"""Exact-equivalence and matched-budget physical pruning tests (issues #4, #13)."""

import os
import tempfile
import unittest
from copy import deepcopy

import torch

from helpers import tiny_model, tiny_input, zero_channel_group
from shared.nerv_targets import get_channel_group_layers
from shared.accounting import param_accounting
from shared.physical_pruning import (
    compute_exact_plan, compute_budget_plan, apply_prune_plan,
    _decoder_head_params_for_keeps, save_pruned_artifact, load_pruned_artifact,
)


class TestExactPruning(unittest.TestCase):
    def test_dead_channel_removed_with_identical_output(self):
        model = tiny_model(seed=5)
        model.eval()
        layers = get_channel_group_layers(model, 'conv')
        zero_channel_group(layers[0], 1)
        zero_channel_group(layers[1], 0)

        plans, head_keep = compute_exact_plan(model, group_thr=1e-4)
        self.assertEqual(len(plans[0].keep_channels), layers[0].channels - 1)
        self.assertNotIn(1, plans[0].keep_channels)
        self.assertEqual(len(plans[1].keep_channels), layers[1].channels - 1)

        pruned = apply_prune_plan(deepcopy(model), plans, head_keep)
        pruned.eval()
        x = tiny_input(batch=3)
        with torch.no_grad():
            out_d, _, _ = model(x)
            out_p, _, _ = pruned(x)
        self.assertLess(float((out_d - out_p).abs().max()), 1e-6)

    def test_param_count_matches_width_propagation(self):
        model = tiny_model(seed=5)
        layers = get_channel_group_layers(model, 'conv')
        zero_channel_group(layers[0], 1)
        plans, head_keep = compute_exact_plan(model, group_thr=1e-4)
        pruned = apply_prune_plan(deepcopy(model), plans, head_keep)

        expected = _decoder_head_params_for_keeps(
            model, layers, [len(p.keep_channels) for p in plans])
        actual = param_accounting(pruned)['decoder_head_params']
        self.assertEqual(actual, expected)
        self.assertLess(actual, param_accounting(model)['decoder_head_params'])

    def test_noop_on_dense_model(self):
        model = tiny_model(seed=6)
        model.eval()
        plans, head_keep = compute_exact_plan(model, group_thr=1e-4)
        for p in plans:
            self.assertEqual(len(p.keep_channels), p.channels_before)
        pruned = apply_prune_plan(deepcopy(model), plans, head_keep)
        x = tiny_input()
        with torch.no_grad():
            out_d, _, _ = model(x)
            out_p, _, _ = pruned(x)
        self.assertEqual(float((out_d - out_p).abs().max()), 0.0)


class TestBudgetPruning(unittest.TestCase):
    def test_budget_is_met_and_min_keep_respected(self):
        model = tiny_model(seed=7)
        full = param_accounting(model)['decoder_head_params']
        plans, head_keep, info = compute_budget_plan(model, budget_reduce=0.5, min_keep=1)

        pruned = apply_prune_plan(deepcopy(model), plans, head_keep)
        achieved = param_accounting(pruned)['decoder_head_params']
        self.assertEqual(achieved, info['achieved_decoder_head_params'])
        self.assertLessEqual(achieved, info['requested_budget_params'])
        self.assertTrue(info['budget_reached'])
        self.assertEqual(info['full_decoder_head_params'], full)
        for p in plans:
            self.assertGreaterEqual(len(p.keep_channels), 1)

        # the pruned model must still run
        with torch.no_grad():
            out, _, _ = pruned(tiny_input())
        self.assertEqual(tuple(out.shape[-2:]), (8, 8))

    def test_lowest_norm_groups_are_removed_first(self):
        model = tiny_model(seed=8)
        layers = get_channel_group_layers(model, 'conv')
        # make one specific group clearly the weakest
        r2 = layers[0].r ** 2
        with torch.no_grad():
            layers[0].conv.weight.data[2 * r2:3 * r2] *= 1e-3
            layers[0].conv.bias.data[2 * r2:3 * r2] *= 1e-3
        full = param_accounting(model)['decoder_head_params']
        plans, _, info = compute_budget_plan(model, budget_params=full - 1, min_keep=1)
        self.assertEqual(info['groups_removed'], 1)
        self.assertNotIn(2, plans[0].keep_channels)

    def test_unreachable_budget_prunes_to_floor(self):
        model = tiny_model(seed=9)
        plans, _, info = compute_budget_plan(model, budget_params=1, min_keep=1)
        for p in plans:
            self.assertEqual(len(p.keep_channels), 1)
        self.assertFalse(info['budget_reached'])


class TestArtifactRoundTrip(unittest.TestCase):
    def test_save_load_gives_identical_model(self):
        model = tiny_model(seed=10)
        model.eval()
        layers = get_channel_group_layers(model, 'conv')
        zero_channel_group(layers[0], 0)
        plans, head_keep = compute_exact_plan(model, group_thr=1e-4)
        pruned = apply_prune_plan(deepcopy(model), plans, head_keep)
        pruned.eval()

        config = {
            'embed': 'pe_1.25_4', 'ks': '0_3_3', 'num_blks': '1_1',
            'enc_strds': [], 'enc_dim': '64_16', 'dec_strds': [2, 2],
            'fc_hw': '2_2', 'reduce': 1.2, 'lower_width': 4,
            'conv_type': ['convnext', 'pshuffel'], 'norm': 'none', 'act': 'gelu',
            'out_bias': 'tanh', 'fc_dim': 8, 'modelsize': 0.01,
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'artifact.pth')
            save_pruned_artifact(path, pruned, plans, head_keep, config)
            reloaded, payload = load_pruned_artifact(path)
            reloaded.eval()
            x = tiny_input(batch=2)
            with torch.no_grad():
                o1, _, _ = pruned(x)
                o2, _, _ = reloaded(x)
            self.assertEqual(float((o1 - o2).abs().max()), 0.0)
            self.assertEqual(payload['format'], 'pcd-nerv-pruned-v2')


if __name__ == '__main__':
    unittest.main()
