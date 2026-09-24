import unittest

import torch

from helpers import tiny_hnerv
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import build_prune_plan_from_keep_sets
from shared.progressive_pruning import apply_progressive_prune_plan


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
        self.assertTrue(all(p.device.type == torch.device(device).type for p in model.parameters()))
        self.assertTrue(all(p.dtype == dtype for p in model.parameters()))
        refreshed = get_channel_group_layers(model)
        self.assertEqual(refreshed[0].channels, layers[0].channels - 1)
        self.assertTrue(torch.isfinite(refreshed[0].group_norms()).all())

    def test_cpu_device_dtype_and_refresh(self):
        self._exercise('cpu', torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_device_preserved(self):
        self._exercise('cuda', torch.float32)


if __name__ == '__main__':
    unittest.main()
