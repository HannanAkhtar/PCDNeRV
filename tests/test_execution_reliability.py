import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from calibrate_compute_pilot_budget import summarize_calibration
from shared.progressive_pruning import load_pilot_checkpoint
from shared.wallclock import (
    fraction_checkpoint_due,
    next_fraction_checkpoint,
)
from train_compute_pilot import (
    _write_csv_if_changed,
    parse_args,
    run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_pilot_args(output, *, eval_every=0, resume=""):
    argv = [
        "--method", "d_start",
        "--budget_seconds", "0.000001",
        "--kappa", "0.5",
        "--outf", str(output),
        "--data_path", str(REPO_ROOT / "data" / "bunny"),
        "--device", "cpu",
        "--max_frames", "1",
        "--crop_list", "8_8",
        "--batchSize", "1",
        "--workers", "0",
        "--enc_strds", "2",
        "--enc_dim", "4_2",
        "--dec_strds", "2",
        "--fc_hw", "4_4",
        "--ks", "1_1_3",
        "--reduce", "2",
        "--modelsize", "0.01",
        "--lower_width", "2",
        "--conv_type", "conv", "pshuffel",
        "--num_blks", "1_1",
        "--max_epochs", "2",
        "--eval_every", str(eval_every),
        "--checkpoint_every_fraction", "0.10",
        "--no_final_fps",
    ]
    if resume:
        argv.extend(["--resume", str(resume)])
    return parse_args(argv)


class RecordingEvaluator:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def __call__(self, model, *args, **kwargs):
        self.calls.append({
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        })
        if self.fail:
            raise RuntimeError("injected evaluator failure")
        call_number = len(self.calls)
        return {
            "PSNR_dB": float(call_number),
            "MS_SSIM": 0.5 + call_number / 100.0,
            "MS_SSIM_device": kwargs["msssim_device"],
            "embeddings": [torch.zeros(1)],
            "frame_count": kwargs["expected_frames"],
            "resolution": "8x8",
        }


class ExecutionReliabilityTests(unittest.TestCase):
    def test_prune_equivalence_cli_default(self):
        with tempfile.TemporaryDirectory() as directory:
            args = tiny_pilot_args(Path(directory) / "run")
        self.assertEqual(args.prune_equivalence_atol, 5e-4)

    def test_checkpoint_fraction_schedule(self):
        boundary = next_fraction_checkpoint(0.0, 0.10)
        self.assertAlmostEqual(boundary, 0.10)
        self.assertFalse(fraction_checkpoint_due(0.099, boundary))
        self.assertTrue(fraction_checkpoint_due(0.10, boundary))
        self.assertAlmostEqual(next_fraction_checkpoint(0.31, 0.10), 0.40)

    def test_history_csv_is_not_rewritten_without_new_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            rows = [{"epoch": 1}]
            count, wrote = _write_csv_if_changed(path, rows, -1)
            self.assertTrue(wrote)
            with mock.patch("train_compute_pilot._write_csv") as writer:
                count, wrote = _write_csv_if_changed(path, rows, count)
                self.assertFalse(wrote)
                writer.assert_not_called()
                _write_csv_if_changed(path, rows + [{"epoch": 2}], count)
                writer.assert_called_once()

    def test_calibration_scales_measured_mean(self):
        result = summarize_calibration([1, 2, 3, 4, 5], target_epochs=300)
        self.assertEqual(result["mean_active_epoch_seconds"], 3.0)
        self.assertEqual(result["recommended_W_seconds"], 900.0)
        self.assertEqual(result["estimated_11_run_counted_training_hours"], 2.75)

    def test_eval_zero_is_final_only_and_uses_checkpointed_final_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            evaluator = RecordingEvaluator()
            result = run(tiny_pilot_args(output), quality_evaluator=evaluator)
            self.assertEqual(len(evaluator.calls), 1)
            self.assertEqual(result["PSNR"], 1.0)
            model, _, _, _ = load_pilot_checkpoint(
                output / "pilot_latest.pth", restore_rng=False
            )
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), evaluator.calls[0][name]))

    def test_periodic_metric_cannot_become_final_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = RecordingEvaluator()
            result = run(
                tiny_pilot_args(Path(directory) / "run", eval_every=1),
                quality_evaluator=evaluator,
            )
            self.assertEqual(len(evaluator.calls), 2)
            self.assertEqual(result["PSNR"], 2.0)
            self.assertEqual(result["MS_SSIM"], 0.52)

    def test_checkpoint_precedes_failure_and_exhausted_resume_does_no_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            checkpoint = output / "pilot_latest.pth"
            with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
                run(
                    tiny_pilot_args(output),
                    quality_evaluator=RecordingEvaluator(fail=True),
                )
            self.assertTrue(checkpoint.is_file())
            _, _, _, before = load_pilot_checkpoint(checkpoint, restore_rng=False)
            evaluator = RecordingEvaluator()
            result = run(
                tiny_pilot_args(output, resume=checkpoint),
                quality_evaluator=evaluator,
            )
            self.assertEqual(result["optimizer_steps"], before["optimizer_steps"])
            self.assertEqual(len(evaluator.calls), 1)


if __name__ == "__main__":
    unittest.main()
