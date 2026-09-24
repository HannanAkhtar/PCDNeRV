import unittest

import torch

from helpers import tiny_hnerv
from shared.nerv_targets import get_channel_group_layers
from shared.physical_pruning import build_prune_plan_from_keep_sets
from shared.progressive_pruning import apply_progressive_prune_plan, assert_optimizer_state_shapes


class OptimizerStatePruningTests(unittest.TestCase):
    def test_adam_moments_are_sliced_and_next_step_runs(self):
        model, image, embedding = tiny_hnerv(fc_dim=12)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            model(image)[0].square().mean().backward()
            optimizer.step()
        layers = get_channel_group_layers(model)
        keep_sets = [list(range(layer.channels)) for layer in layers]
        keep_sets[0].pop(0)
        plans, head = build_prune_plan_from_keep_sets(model, keep_sets)
        old_states = {
            parameter: {key: value.clone() if torch.is_tensor(value) else value
                        for key, value in state.items()}
            for parameter, state in optimizer.state.items()
        }
        result = apply_progressive_prune_plan(
            model, optimizer, plans, head, verify_embedding=embedding
        )
        for replacement in result['replacements']:
            before = old_states.get(replacement.old_parameter, {})
            after = optimizer.state[replacement.new_parameter]
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in before:
                    expected = replacement.slice_tensor(before[key])
                    self.assertTrue(torch.equal(after[key], expected), (replacement.name, key))
        assert_optimizer_state_shapes(optimizer)
        optimizer.zero_grad()
        model(image)[0].square().mean().backward()
        optimizer.step()


if __name__ == '__main__':
    unittest.main()
