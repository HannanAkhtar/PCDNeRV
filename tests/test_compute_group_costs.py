import unittest

from helpers import tiny_hnerv
from shared.compute_pruning import compute_group_mac_costs, verify_one_group_cost


class ComputeGroupCostTests(unittest.TestCase):
    def test_positive_normalized_and_matches_actual_prune(self):
        model, _, embedding = tiny_hnerv(fc_dim=12)
        costs = compute_group_mac_costs(model, embedding)
        self.assertTrue(all(row['raw_MACs'] > 0 for row in costs.values()))
        weighted_mean = sum(row['normalized'] * row['groups'] for row in costs.values()) / sum(
            row['groups'] for row in costs.values()
        )
        self.assertAlmostEqual(weighted_mean, 1.0, places=12)
        check = verify_one_group_cost(model, embedding)
        self.assertTrue(check['matches'], check)


if __name__ == '__main__':
    unittest.main()
