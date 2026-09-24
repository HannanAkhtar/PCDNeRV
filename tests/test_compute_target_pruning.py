import unittest

from helpers import tiny_hnerv
from shared.accounting import measure_decoder_compute
from shared.compute_pruning import compute_target_prune_plan
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import apply_prune_plan, compute_budget_plan


class ComputeTargetPruningTests(unittest.TestCase):
    def test_target_minimum_width_and_measured_accounting(self):
        model, _, embedding = tiny_hnerv(fc_dim=16)
        start = measure_decoder_compute(model, embedding)['total_MACs']
        plans, head, info = compute_target_prune_plan(
            model, embedding, kappa=0.25, start_macs=start, min_keep=1
        )
        self.assertTrue(info['target_reached'])
        self.assertTrue(info['MACs_monotonic'])
        apply_prune_plan(model, plans, head)
        actual = measure_decoder_compute(model, embedding)['total_MACs']
        self.assertEqual(actual, info['achieved_MACs'])
        self.assertAlmostEqual(info['achieved_kappa'], 1 - actual / start)
        self.assertTrue(all(layer.channels >= 1 for layer in get_channel_group_layers(model)))

    def test_compute_and_parameter_planners_are_distinct(self):
        model, _, embedding = tiny_hnerv(fc_dim=16)
        c_plans, _, c_info = compute_target_prune_plan(model, embedding, kappa=0.2)
        p_plans, _, p_info = compute_budget_plan(model, budget_reduce=0.2)
        self.assertIn('target_MACs', c_info)
        self.assertIn('requested_budget_params', p_info)
        self.assertNotIn('requested_budget_params', c_info)
        self.assertNotIn('target_MACs', p_info)
        self.assertTrue(c_plans and p_plans)


if __name__ == '__main__':
    unittest.main()
