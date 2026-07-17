"""
Method-specific training-step tests, run through the REAL train_one_epoch —
including the lambda-sweep regression test (deployment-team issue #7).
"""

import types
import unittest

import torch

from helpers import tiny_model, tiny_args, tiny_input
from model_all import TransformInput
from shared.groups import group_lasso_loss
from shared.nerv_targets import get_channel_group_layers
from shared.pcd_solver import PCDSolver
from train_pcd_nerv import train_one_epoch
from hnerv_utils import loss_fn


def make_train_args(**overrides):
    args = tiny_args()
    args.vid = 'unittest'
    args.loss = 'L2'
    args.debug = False
    args.print_freq = 10_000
    args.epochs = 5
    args.lr = 0.01
    args.lr_type = 'cosine_0.1_1_0.1'
    for k, v in overrides.items():
        setattr(args, k, v)
    args.transform_func = TransformInput(args)
    return args


def fake_loader(n_batches=2, seed=0):
    torch.manual_seed(seed)
    return [
        {'img': torch.rand(1, 3, 8, 8), 'norm_idx': torch.tensor([0.3 + 0.4 * i]),
         'idx': torch.tensor([i])}
        for i in range(n_batches)
    ]


def run_one_epoch(model, method, lambda_prox=1e-3, ws_weight=1e-5, args=None,
                  is_warmup=False):
    args = args or make_train_args()
    layers = get_channel_group_layers(model, 'conv')
    opt = torch.optim.Adam(model.parameters())
    solver = PCDSolver(tau=0.1) if method == 'pcd' else None
    return train_one_epoch(
        model, fake_loader(), opt, args, method, layers, solver,
        lambda_prox, ws_weight, epoch=0, is_warmup=is_warmup,
        device='cpu', plog=lambda msg: None)


def total_group_norm(model):
    return float(group_lasso_loss(get_channel_group_layers(model, 'conv')))


class TestLambdaSweepCorrectness(unittest.TestCase):
    def test_passed_lambda_wins_over_args_attribute(self):
        """Issue #7 regression: the update must use the RUN-specific lambda,
        never args.lambda_prox. We plant a contradictory decoy on args."""
        # decoy says huge, passed value is 0 -> nothing must be shrunk to zero
        m1 = tiny_model(seed=3)
        args1 = make_train_args()
        args1.lambda_prox = 1e9  # decoy
        run_one_epoch(m1, 'prox_gl', lambda_prox=0.0, args=args1)
        self.assertGreater(total_group_norm(m1), 0.5)

        # decoy says 0, passed value is huge -> every group must die
        m2 = tiny_model(seed=3)
        args2 = make_train_args()
        args2.lambda_prox = 0.0  # decoy
        run_one_epoch(m2, 'prox_gl', lambda_prox=1e9, args=args2)
        self.assertEqual(total_group_norm(m2), 0.0)


class TestMethodSteps(unittest.TestCase):
    def test_baseline_never_applies_prox(self):
        m = tiny_model(seed=1)
        trm = run_one_epoch(m, 'baseline', lambda_prox=1e9)
        self.assertGreater(total_group_norm(m), 0.5)
        self.assertEqual(trm['effective_step'], 'baseline')
        self.assertNotIn('mu', trm)

    def test_warmup_turns_any_method_into_baseline(self):
        m = tiny_model(seed=1)
        trm = run_one_epoch(m, 'pcd', lambda_prox=1e9, is_warmup=True)
        self.assertGreater(total_group_norm(m), 0.5)
        self.assertEqual(trm['effective_step'], 'baseline')
        self.assertNotIn('mu', trm)

    def test_prox_gl_kills_groups_with_large_lambda(self):
        m = tiny_model(seed=1)
        trm = run_one_epoch(m, 'prox_gl', lambda_prox=1e9)
        self.assertEqual(total_group_norm(m), 0.0)
        self.assertNotIn('mu', trm)  # no PCD diagnostics outside pcd

    def test_pcd_runs_solver_and_prox(self):
        m = tiny_model(seed=1)
        trm = run_one_epoch(m, 'pcd', lambda_prox=1e9)
        self.assertEqual(total_group_norm(m), 0.0)
        for key in ('mu', 'conflict', 'cosine_sim', 'primary_efficiency', 'tau'):
            self.assertIn(key, trm)

    def test_weighted_sum_gradient_composition(self):
        """d(L_rec + w*L_gl)/dp == d(L_rec)/dp + w * d(L_gl)/dp."""
        w = 0.37
        m_total = tiny_model(seed=2)
        m_rec = tiny_model(seed=2)
        m_gl = tiny_model(seed=2)
        x = tiny_input()
        target = torch.rand(1, 3, 8, 8)

        out, _, _ = m_total(x)
        layers_t = get_channel_group_layers(m_total, 'conv')
        (loss_fn(out, target, 'L2') + w * group_lasso_loss(layers_t)).backward()

        out_r, _, _ = m_rec(x)
        loss_fn(out_r, target, 'L2').backward()
        group_lasso_loss(get_channel_group_layers(m_gl, 'conv')).backward()

        for (n1, p1), (_, p2), (_, p3) in zip(
                m_total.named_parameters(), m_rec.named_parameters(), m_gl.named_parameters()):
            g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
            g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
            g3 = p3.grad if p3.grad is not None else torch.zeros_like(p3)
            self.assertTrue(torch.allclose(g1, g2 + w * g3, atol=1e-6), f'mismatch at {n1}')

    def test_weighted_sum_logs_total_loss(self):
        m = tiny_model(seed=1)
        trm = run_one_epoch(m, 'weighted_sum', ws_weight=0.1)
        self.assertIn('total_loss', trm)
        self.assertGreaterEqual(trm['total_loss'], trm['primary_loss'])


if __name__ == '__main__':
    unittest.main()
