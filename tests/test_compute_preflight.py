import unittest

from helpers import tiny_hnerv_config, tiny_hnerv
from preflight_compute_pilot import profile_configuration


class ComputePreflightTests(unittest.TestCase):
    def test_reports_pruning_and_dense_search_without_training(self):
        _, image, _ = tiny_hnerv(seed=2, fc_dim=12)
        config = tiny_hnerv_config(fc_dim=12)
        config["modelsize"] = 0.02
        report = profile_configuration(
            config,
            image,
            frame_count=4,
            kappas=(0.20,),
            candidate_model_sizes=(0.005, 0.01, 0.015),
        )

        self.assertFalse(report["training_performed"])
        self.assertGreater(report["dense"]["decoder_MACs"], 0)
        self.assertEqual(len(report["targets"]), 1)
        target = report["targets"][0]
        self.assertIn("planning_wallclock_seconds", target["hard_compute_pruning"])
        self.assertIn("groups_removed", target["hard_compute_pruning"])
        self.assertIn("final_widths", target["hard_compute_pruning"])
        self.assertIn("actual_decoder_head_params", target["d_small_compute"])
        self.assertIn("decoder_GFLOPs", target["d_small_compute"])
        self.assertIn("relative_compute_target_error", target["d_small_compute"])


if __name__ == "__main__":
    unittest.main()
