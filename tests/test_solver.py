"""PCDSolver constraint tests (active / inactive) against the solver math."""

import math
import unittest

import torch

from helpers import REPO_ROOT  # noqa: F401  (sys.path bootstrap)
from shared.pcd_solver import PCDSolver


def _norm_sq(gs):
    return sum(float(g.pow(2).sum()) for g in gs)


class TestSolverConstraint(unittest.TestCase):
    def test_inactive_when_aligned(self):
        """g_gl parallel to g_primary: constraint satisfied, pure primary step."""
        torch.manual_seed(0)
        g_p = [torch.randn(5, 3), torch.randn(7)]
        g_gl = [2.0 * g for g in g_p]
        solver = PCDSolver(tau=0.5)
        combined, diag = solver.step(g_p, g_gl)
        self.assertFalse(diag['conflict'])
        self.assertEqual(diag['mu'], 0.0)
        self.assertAlmostEqual(diag['cosine_sim'], 1.0, places=5)
        self.assertEqual(diag['ce_efficiency'], 1.0)
        for c, g in zip(combined, g_p):
            self.assertTrue(torch.equal(c, g))

    def test_active_when_opposed(self):
        """g_gl = -g_primary: mu matches the closed form and the normalized
        constraint g~_gl . d~ = tau * ||g~_gl||^2 holds with equality."""
        torch.manual_seed(1)
        tau, eps = 0.3, 1e-8
        g_p = [torch.randn(4, 4)]
        g_gl = [-g_p[0].clone()]
        solver = PCDSolver(tau=tau, eps=eps)
        combined, diag = solver.step(g_p, g_gl)
        self.assertTrue(diag['conflict'])

        # at t=1 the EMA bias correction cancels: s_x = 1/(||g_x|| + eps)
        n_p = math.sqrt(_norm_sq(g_p))
        n_g = math.sqrt(_norm_sq(g_gl))
        s_p, s_g = 1 / (n_p + eps), 1 / (n_g + eps)
        dot = float(sum((a * b).sum() for a, b in zip(g_p, g_gl)))
        t12 = s_p * s_g * dot
        t22 = s_g * s_g * _norm_sq(g_gl)
        mu_expected = (tau * t22 - t12) / t22
        self.assertAlmostEqual(diag['mu'], mu_expected, places=5)

        # rebuild the pre-rescale direction and check the equality constraint
        d_normed = [s_p * gp + diag['mu'] * s_g * gg for gp, gg in zip(g_p, g_gl)]
        lhs = float(sum((s_g * gg * d).sum() for gg, d in zip(g_gl, d_normed)))
        self.assertAlmostEqual(lhs, tau * t22, places=5)

        # the returned direction is rescaled to keep the raw primary magnitude
        n_combined = math.sqrt(_norm_sq(combined))
        self.assertAlmostEqual(n_combined, n_p, places=4)
        self.assertLess(diag['ce_efficiency'], 1.0)

    def test_degenerate_zero_gl_gradient(self):
        g_p = [torch.randn(3, 3)]
        g_gl = [torch.zeros(3, 3)]
        solver = PCDSolver(tau=0.9)
        combined, diag = solver.step(g_p, g_gl)
        self.assertFalse(diag['conflict'])
        self.assertTrue(torch.equal(combined[0], g_p[0]))


if __name__ == '__main__':
    unittest.main()
