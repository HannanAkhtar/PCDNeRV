import argparse
import unittest

from train_compute_pilot import _validate_resume_configuration


class ComputePilotResumeValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "arch": "hnerv", "crop_list": "8_8", "resize_list": "-1",
            "embed": "", "enc_strds": [2], "enc_dim": "4_2",
            "dec_strds": [2], "fc_hw": "4_4", "fc_dim": 4,
            "ks": "0_1_3", "reduce": 1.2, "lower_width": 2,
            "num_blks": "1_1", "conv_type": ["convnext", "pshuffel"],
            "norm": "none", "act": "gelu", "out_bias": "tanh",
            "modelsize": 0.01, "saturate_stages": -1,
        }
        self.args = argparse.Namespace(
            method="m4_pcd", kappa=0.5, manualSeed=7, budget_seconds=100.0
        )
        self.payload = {
            "method": "m4_pcd", "kappa": 0.5, "seed": 7,
            "budget_seconds": 100.0, "epoch": 1, "history": [{"epoch": 1}],
            "selection": {"base_config": dict(self.base)},
        }

    def test_matching_resume_is_accepted(self):
        _validate_resume_configuration(self.payload, self.args, self.base)

    def test_every_scientific_identity_mismatch_is_reported(self):
        args = argparse.Namespace(
            method="m3_group_lasso", kappa=0.7, manualSeed=8,
            budget_seconds=120.0,
        )
        changed_base = dict(self.base, lower_width=3)
        with self.assertRaisesRegex(ValueError, "method") as caught:
            _validate_resume_configuration(self.payload, args, changed_base)
        message = str(caught.exception)
        for field in ("kappa", "seed", "budget_seconds", "architecture"):
            self.assertIn(field, message)

    def test_legacy_checkpoint_without_history_or_selection_fails(self):
        payload = dict(self.payload)
        payload.pop("history")
        payload.pop("selection")
        with self.assertRaisesRegex(ValueError, "complete training history"):
            _validate_resume_configuration(payload, self.args, self.base)


if __name__ == "__main__":
    unittest.main()
