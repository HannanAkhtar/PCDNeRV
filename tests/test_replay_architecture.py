import tempfile
import unittest
from pathlib import Path

import torch

from helpers import tiny_hnerv, tiny_hnerv_config
from shared.accounting import decoder_layer_widths, measure_decoder_compute, param_accounting
from shared.compute_pruning import build_random_architecture_from_pruned_artifact
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import apply_prune_plan, build_prune_plan_from_keep_sets, save_pruned_artifact


class ReplayArchitectureTests(unittest.TestCase):
    def test_exact_architecture_but_not_weights(self):
        source, _, embedding = tiny_hnerv(seed=2, fc_dim=12)
        layers = get_channel_group_layers(source)
        keep_sets = [list(range(layer.channels)) for layer in layers]
        keep_sets[0].pop(0)
        plans, head = build_prune_plan_from_keep_sets(source, keep_sets)
        apply_prune_plan(source, plans, head)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'source_pruned.pth'
            save_pruned_artifact(artifact, source, plans, head, tiny_hnerv_config(fc_dim=12))
            replay, _, details = build_random_architecture_from_pruned_artifact(
                artifact, seed=99, sample_embedding=embedding
            )
        self.assertEqual(decoder_layer_widths(replay), decoder_layer_widths(source))
        self.assertEqual(
            param_accounting(replay)['decoder_head_params'],
            param_accounting(source)['decoder_head_params'],
        )
        self.assertEqual(
            measure_decoder_compute(replay, embedding)['total_MACs'],
            measure_decoder_compute(source, embedding)['total_MACs'],
        )
        self.assertTrue(any(
            not torch.equal(value, replay.state_dict()[name])
            for name, value in source.state_dict().items()
        ))
        self.assertEqual(details['decoder_MACs'], measure_decoder_compute(source, embedding)['total_MACs'])


if __name__ == '__main__':
    unittest.main()
