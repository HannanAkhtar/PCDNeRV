import unittest

from shared.wallclock import ActiveTrainingBudget, learning_rate_at_fraction


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class WallClockBudgetTests(unittest.TestCase):
    def test_only_measured_intervals_count_and_stop_after_step(self):
        clock = FakeClock()
        budget = ActiveTrainingBudget(1.0, time_fn=clock)
        clock.advance(20.0)  # excluded evaluation/checkpoint interval
        self.assertEqual(budget.fraction, 0.0)

        def first_step():
            clock.advance(0.6)
            return 'done'

        value, stop = budget.completed_optimizer_step(first_step)
        self.assertEqual(value, 'done')
        self.assertFalse(stop)
        self.assertAlmostEqual(budget.fraction, 0.6)

        def crossing_step():
            clock.advance(0.5)
            return 'completed-before-stop'

        value, stop = budget.completed_optimizer_step(crossing_step)
        self.assertEqual(value, 'completed-before-stop')
        self.assertTrue(stop)
        self.assertAlmostEqual(budget.counted_training_seconds, 1.1)

    def test_lr_uses_budget_fraction(self):
        self.assertAlmostEqual(
            learning_rate_at_fraction(1e-3, 'cosine_0.1_1_0.1', 0.0), 1e-4
        )
        self.assertAlmostEqual(
            learning_rate_at_fraction(1e-3, 'cosine_0.1_1_0.1', 1.0), 0.0,
            places=12,
        )


if __name__ == '__main__':
    unittest.main()
