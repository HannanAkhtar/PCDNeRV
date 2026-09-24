import unittest

import torch

from helpers import tiny_hnerv_config
from shared.accounting import measure_decoder_compute
from shared.compute_pruning import derive_hnerv_config_for_modelsize, find_compute_matched_hnerv
from shared.physical_pruning import build_model_from_config


class ComputeMatchedDenseTests(unittest.TestCase):
    def test_search_uses_actual_compute_and_returns_closest(self):
        base = tiny_hnerv_config(fc_dim=20)
        base['modelsize'] = 0.01
        image = torch.randn(1, 3, 8, 8)
        exact_config = derive_hnerv_config_for_modelsize(
            base, 0.007, frame_count=8, output_hw=(8, 8)
        )
        exact_model = build_model_from_config(exact_config)
        target = measure_decoder_compute(exact_model, exact_model(image)[1][0])['total_MACs']
        model, config, info = find_compute_matched_hnerv(
            base, target, image, frame_count=8,
            candidate_model_sizes=[0.004, 0.007, 0.009], tolerance=0.02,
        )
        actual = measure_decoder_compute(model, model(image)[1][0])['total_MACs']
        self.assertEqual(actual, info['actual_MACs'])
        self.assertEqual(actual, target)
        self.assertTrue(info['within_tolerance'])
        self.assertAlmostEqual(config['modelsize'], 0.007)


if __name__ == '__main__':
    unittest.main()
