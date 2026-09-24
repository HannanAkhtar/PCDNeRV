import json
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate_saved_models import (
    ModelSpec,
    build_run_model,
    discover_models,
    evaluate_quality_and_embeddings,
)
from shared.accounting import benchmark_decoder_cuda, measure_decoder_compute, param_accounting
from shared.physical_pruning import (
    build_model_from_config,
    compute_exact_plan,
    save_pruned_artifact,
)


def tiny_config():
    return {
        "method": "baseline",
        "arch": "hnerv",
        "vid": "bunny_smoke",
        "data_path": "data/bunny_smoke",
        "data_split": "1_1_1",
        "shuffle_data": False,
        "crop_list": "192_384",
        "resize_list": "-1",
        "embed": "",
        "enc_strds": [2],
        "enc_dim": "4_2",
        "dec_strds": [2],
        "fc_hw": "1_1",
        "fc_dim": 4,
        "ks": "1_1_3",
        "reduce": 2.0,
        "lower_width": 2,
        "num_blks": "1_1",
        "conv_type": ["conv", "pshuffel"],
        "norm": "none",
        "act": "gelu",
        "out_bias": "tanh",
        "modelsize": 0.01,
    }


class UnifiedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.run_dir = cls.root / "training" / "main_grid" / "bunny" / "run"
        (cls.run_dir / "results").mkdir(parents=True)
        (cls.run_dir / "pruned").mkdir()
        cls.config = tiny_config()

        torch.manual_seed(3)
        cls.model = build_model_from_config(cls.config)
        cls.tag = "baseline_seed1"
        cls.checkpoint = cls.run_dir / f"{cls.tag}_model_latest.pth"
        torch.save({"state_dict": cls.model.state_dict(), "config": cls.config}, cls.checkpoint)
        cls.result_json = cls.run_dir / "results" / f"{cls.tag}.json"
        cls.result_json.write_text(
            json.dumps({"config": cls.config, "final": {"pred_seen_psnr": 0.0}, "epochs": []}),
            encoding="utf-8",
        )

        plans, head_keep = compute_exact_plan(cls.model, group_thr=0.0)
        cls.artifact = cls.run_dir / "pruned" / f"{cls.tag}_exact_pruned.pth"
        save_pruned_artifact(str(cls.artifact), cls.model, plans, head_keep, cls.config)

        cls.recovery_dir = cls.root / "recovery" / cls.tag / "budget50" / "bunny" / "run"
        (cls.recovery_dir / "results").mkdir(parents=True)
        recovery_config = dict(cls.config)
        recovery_config.update({"method": "baseline", "init_artifact": str(cls.artifact)})
        cls.recovery_checkpoint = cls.recovery_dir / "recovery_seed1_model_latest.pth"
        torch.save({"state_dict": cls.model.state_dict(), "config": recovery_config}, cls.recovery_checkpoint)
        cls.recovery_json = cls.recovery_dir / "results" / "recovery_seed1.json"
        cls.recovery_json.write_text(
            json.dumps({"config": recovery_config, "final": {}, "epochs": []}),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_dense_model_loading(self):
        loaded = build_run_model(
            ModelSpec(
                model_type="dense_checkpoint", checkpoint=str(self.checkpoint),
                result_json=str(self.result_json), tag=self.tag,
            ),
            self.root,
        )
        self.assertEqual(loaded.model_type, "dense_checkpoint")
        self.assertEqual(set(loaded.model.state_dict()), set(self.model.state_dict()))

    def test_auto_discovery_finds_all_loading_modes(self):
        specs = discover_models(self.root)
        types = {spec.model_type for spec in specs}
        self.assertIn("dense_checkpoint", types)
        self.assertIn("pruned_artifact", types)
        self.assertIn("recovery_checkpoint", types)

    def test_pruned_artifact_loading(self):
        loaded = build_run_model(
            ModelSpec(model_type="pruned_artifact", artifact=str(self.artifact), tag=self.tag),
            self.root,
        )
        self.assertEqual(loaded.model_type, "pruned_artifact")
        self.assertEqual(set(loaded.model.state_dict()), set(self.model.state_dict()))

    def test_recovery_checkpoint_reconstruction(self):
        loaded = build_run_model(
            ModelSpec(
                model_type="recovery_checkpoint", checkpoint=str(self.recovery_checkpoint),
                result_json=str(self.recovery_json), tag="recovery_seed1",
            ),
            self.root,
        )
        self.assertEqual(loaded.model_type, "recovery_checkpoint")
        self.assertEqual(Path(loaded.source_artifact), self.artifact.resolve())

    def test_parameter_accounting(self):
        embeddings = [torch.zeros(1, 2, 96, 192), torch.zeros(1, 2, 96, 192)]
        result = param_accounting(self.model, embeddings)
        expected_decoder = sum(
            p.numel() for name, p in self.model.named_parameters()
            if p.requires_grad and (name.startswith("decoder.") or name.startswith("head_layer."))
        )
        self.assertEqual(result["decoder_head_params"], expected_decoder)
        self.assertEqual(result["embedding_storage_values"], 2 * embeddings[0].numel())
        self.assertEqual(
            result["total_stored_representation_values"],
            expected_decoder + 2 * embeddings[0].numel(),
        )

    def test_mac_flop_convention(self):
        embedding = torch.zeros(1, 2, 96, 192)
        result = measure_decoder_compute(self.model.cpu(), embedding)
        self.assertGreater(result["total_MACs"], 0)
        self.assertEqual(result["total_FLOPs"], 2 * result["total_MACs"])
        self.assertAlmostEqual(result["total_GFLOPs"], 2 * result["total_GMACs"], places=12)
        self.assertEqual(sum(x["MACs"] for x in result["per_layer"]), result["total_MACs"])

    def test_cpu_quality_and_flop_fallback(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "bunny"
        result = evaluate_quality_and_embeddings(
            self.model.cpu(), self.config, data_path, device="cpu",
            expected_frames=132, max_frames=1,
            compute_msssim=True, msssim_device="cpu",
        )
        self.assertEqual(result["frame_count"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(result["PSNR_dB"])))
        self.assertTrue(torch.isfinite(torch.tensor(result["MS_SSIM"])))
        self.assertEqual(result["MS_SSIM_device"], "cpu")
        compute = measure_decoder_compute(self.model, result["embeddings"][0])
        self.assertGreater(compute["total_GMACs"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_decoder_timing(self):
        embeddings = [torch.zeros(1, 2, 96, 192), torch.ones(1, 2, 96, 192)]
        result = benchmark_decoder_cuda(self.model, embeddings, warmup=1, iterations=3)
        self.assertGreater(result["latency_mean_ms"], 0)
        self.assertAlmostEqual(result["FPS"], 1000.0 / result["latency_mean_ms"], places=7)
        self.assertEqual(result["precision"], "FP32")
        self.assertEqual(result["batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
