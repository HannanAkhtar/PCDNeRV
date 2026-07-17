"""Channel-group abstraction tests (deployment-team issues #1, #2)."""

import math
import unittest

import torch

from helpers import tiny_model, zero_channel_group
from shared.groups import (
    group_lasso_loss, apply_group_prox, group_sparsity,
    compute_sparsity_report,
)
from shared.nerv_targets import (
    get_channel_group_layers, param_scope, DECODER_HEAD_SCOPES,
)


class TestPixelShuffleGrouping(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.layers = get_channel_group_layers(self.model, 'conv')

    def test_group_count_equals_postshuffle_channels(self):
        """One group per post-shuffle channel: channels * r^2 == conv rows."""
        self.assertTrue(len(self.layers) >= 2)
        for layer in self.layers:
            rows = layer.conv.weight.shape[0]
            self.assertEqual(layer.channels * layer.r ** 2, rows)
            self.assertEqual(layer.r, 2)  # dec_strds [2, 2] with pshuffel

    def test_group_view_covers_all_rows_and_biases(self):
        for layer in self.layers:
            view = layer.group_view()
            w, b = layer.conv.weight, layer.conv.bias
            expected_cols = layer.r ** 2 * w[0].numel() + layer.r ** 2
            self.assertEqual(view.shape, (layer.channels, expected_cols))
            self.assertEqual(view.numel(), w.numel() + b.numel())

    def test_fc_target_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            get_channel_group_layers(self.model, 'fc')


class TestBiasInclusiveNorms(unittest.TestCase):
    def test_norm_matches_manual_weight_plus_bias(self):
        model = tiny_model()
        layer = get_channel_group_layers(model, 'conv')[0]
        norms = layer.group_norms()
        r2 = layer.r ** 2
        for ch in range(layer.channels):
            w_block = layer.conv.weight[ch * r2:(ch + 1) * r2]
            b_block = layer.conv.bias[ch * r2:(ch + 1) * r2]
            manual = math.sqrt(float(w_block.pow(2).sum() + b_block.pow(2).sum()))
            self.assertAlmostEqual(float(norms[ch]), manual, places=5)

    def test_zero_weights_nonzero_bias_is_NOT_dead(self):
        """The v1 failure mode: weights zero but bias alive -> group not dead."""
        model = tiny_model()
        layer = get_channel_group_layers(model, 'conv')[0]
        r2 = layer.r ** 2
        with torch.no_grad():
            layer.conv.weight.data[0:r2] = 0
            layer.conv.bias.data[0:r2] = 0.5
        self.assertGreater(float(layer.group_norms()[0]), 0.1)
        dead, _ = group_sparsity([layer], thr=1e-4)
        self.assertEqual(dead, 0)

    def test_loss_is_sum_of_group_norms_and_differentiable(self):
        model = tiny_model()
        layers = get_channel_group_layers(model, 'conv')
        loss = group_lasso_loss(layers)
        manual = sum(float(l.group_norms().sum()) for l in layers)
        self.assertAlmostEqual(float(loss), manual, places=4)
        loss.backward()
        for layer in layers:
            self.assertIsNotNone(layer.conv.weight.grad)
            self.assertIsNotNone(layer.conv.bias.grad)
            self.assertGreater(float(layer.conv.bias.grad.abs().sum()), 0.0)


class TestProximalShrinkage(unittest.TestCase):
    def test_whole_group_scaled_by_same_factor(self):
        model = tiny_model()
        layer = get_channel_group_layers(model, 'conv')[0]
        r2 = layer.r ** 2
        w_before = layer.conv.weight.data.clone()
        b_before = layer.conv.bias.data.clone()
        norms_before = layer.group_norms().clone()

        thr = 0.5 * float(norms_before.min())
        apply_group_prox([layer], thr)

        for ch in range(layer.channels):
            expected = max(0.0, 1.0 - thr / float(norms_before[ch]))
            sl = slice(ch * r2, (ch + 1) * r2)
            w_ratio = layer.conv.weight.data[sl] / w_before[sl]
            b_ratio = layer.conv.bias.data[sl] / b_before[sl]
            self.assertTrue(torch.allclose(
                w_ratio, torch.full_like(w_ratio, expected), atol=1e-5))
            self.assertTrue(torch.allclose(
                b_ratio, torch.full_like(b_ratio, expected), atol=1e-5))

    def test_below_threshold_group_becomes_exactly_zero(self):
        model = tiny_model()
        layer = get_channel_group_layers(model, 'conv')[0]
        r2 = layer.r ** 2
        with torch.no_grad():
            layer.conv.weight.data[0:r2] *= 1e-6
            layer.conv.bias.data[0:r2] *= 1e-6
        norm0 = float(layer.group_norms()[0])
        apply_group_prox([layer], norm0 * 2)
        self.assertEqual(float(layer.conv.weight.data[0:r2].abs().sum()), 0.0)
        self.assertEqual(float(layer.conv.bias.data[0:r2].abs().sum()), 0.0)
        dead, _ = group_sparsity([layer], thr=1e-8)
        self.assertEqual(dead, 1)


class TestSparsityReport(unittest.TestCase):
    def test_report_counts_channel_groups_and_scopes(self):
        model = tiny_model()
        layers = get_channel_group_layers(model, 'conv')
        zero_channel_group(layers[0], 0)
        rep = compute_sparsity_report(model, layers, param_scope, DECODER_HEAD_SCOPES)
        self.assertEqual(rep['group_total'], sum(l.channels for l in layers))
        self.assertEqual(rep['group_sparse'], 1)
        # NeRV-PE has no encoder: decoder+head scope == all trainable
        self.assertEqual(rep['decoder_head_param_total'], rep['trainable_param_total'])
        # targeted params include the biases of targeted convs
        expected_targeted = sum(
            l.conv.weight.numel() + l.conv.bias.numel() for l in layers)
        self.assertEqual(rep['targeted_param_total'], expected_targeted)


if __name__ == '__main__':
    unittest.main()
