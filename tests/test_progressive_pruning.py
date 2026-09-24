import unittest

import torch

from helpers import tiny_hnerv
from shared.accounting import measure_decoder_compute
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import build_prune_plan_from_keep_sets
from shared.progressive_pruning import (
    apply_progressive_prune_plan,
    verify_prune_equivalence,
)
from train_compute_pilot import _prune_event


class ProgressivePruningTests(unittest.TestCase):
    def _exercise(self, device, dtype):
        model, _, embedding = tiny_hnerv(fc_dim=12, device=device, dtype=dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        layers = get_channel_group_layers(model)
        keep_sets = [list(range(layer.channels)) for layer in layers]
        keep_sets[0].pop(0)
        plans, head = build_prune_plan_from_keep_sets(model, keep_sets)
        result = apply_progressive_prune_plan(
            model, optimizer, plans, head, verify_embedding=embedding
        )
        self.assertLess(result['max_equivalence_error'], 1e-5)
        self.assertLess(result['mean_equivalence_error'], 1e-5)
        self.assertLess(result['rmse_equivalence_error'], 1e-5)
        self.assertTrue(all(p.device.type == torch.device(device).type for p in model.parameters()))
        self.assertTrue(all(p.dtype == dtype for p in model.parameters()))
        refreshed = get_channel_group_layers(model)
        self.assertEqual(refreshed[0].channels, layers[0].channels - 1)
        self.assertTrue(torch.isfinite(refreshed[0].group_norms()).all())

    def test_cpu_device_dtype_and_refresh(self):
        self._exercise('cpu', torch.float64)

    def test_artificial_difference_below_default_tolerance_passes(self):
        zeroed = torch.zeros(2, 3, 4, 4)
        rebuilt = torch.full_like(zeroed, 4e-4)
        statistics = verify_prune_equivalence(zeroed, rebuilt, atol=5e-4)
        self.assertAlmostEqual(statistics['max_equivalence_error'], 4e-4)
        self.assertAlmostEqual(statistics['mean_equivalence_error'], 4e-4)
        self.assertAlmostEqual(statistics['rmse_equivalence_error'], 4e-4)

    def test_clearly_wrong_artificial_difference_fails(self):
        zeroed = torch.zeros(2, 3, 4, 4)
        rebuilt = torch.full_like(zeroed, 1e-2)
        with self.assertRaisesRegex(AssertionError, r"max=0.01"):
            verify_prune_equivalence(zeroed, rebuilt, atol=5e-4)

    def test_pruning_event_exposes_all_equivalence_statistics(self):
        model, _, embedding = tiny_hnerv(fc_dim=12)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_macs = measure_decoder_compute(model, embedding)['total_MACs']
        event = _prune_event(
            model,
            optimizer,
            embedding,
            start_macs=start_macs,
            target_macs=start_macs * 0.9,
            min_keep=1,
            eligible_eps=None,
            allow_overshoot=True,
            equivalence_tol=5e-4,
        )
        self.assertGreater(event['groups_removed'], 0)
        for key in (
            'max_equivalence_error',
            'mean_equivalence_error',
            'rmse_equivalence_error',
        ):
            self.assertIn(key, event)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_device_preserved(self):
        self._exercise('cuda', torch.float32)


if __name__ == '__main__':
    unittest.main()
