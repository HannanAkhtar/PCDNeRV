import unittest

from shared.wallclock import (
    gradual_kappa_target, hard_prune_due, pilot_phase, threshold_removal_active,
)


class PilotScheduleTests(unittest.TestCase):
    def test_m1_waits_until_point_nine(self):
        self.assertEqual(pilot_phase('m1_posthoc', 0.899), 'reconstruction_dense')
        self.assertFalse(hard_prune_due('m1_posthoc', 0.899))
        self.assertTrue(hard_prune_due('m1_posthoc', 0.9))

    def test_m2_cubic_is_monotone_and_reaches_target(self):
        values = [gradual_kappa_target(0.7, value) for value in (0.1, 0.3, 0.5, 0.8)]
        self.assertEqual(values[0], 0.0)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 0.7)

    def test_m3_m4_windows_and_finetune(self):
        for method in ('m3_group_lasso', 'm4_pcd'):
            self.assertFalse(threshold_removal_active(method, 0.099))
            self.assertTrue(threshold_removal_active(method, 0.5))
            self.assertFalse(threshold_removal_active(method, 0.9))
            self.assertTrue(hard_prune_due(method, 0.9))
            self.assertEqual(pilot_phase(method, 0.95), 'reconstruction_finetune')


if __name__ == '__main__':
    unittest.main()
