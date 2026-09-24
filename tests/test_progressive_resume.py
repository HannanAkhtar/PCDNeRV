import tempfile
import unittest
from pathlib import Path

import torch

from helpers import tiny_hnerv, tiny_hnerv_config
from shared.accounting import decoder_layer_widths, measure_decoder_compute
from shared.nerv_targets import get_channel_group_layers
from shared.pcd_solver import PCDSolver
from shared.physical_pruning import build_prune_plan_from_keep_sets
from shared.progressive_pruning import (
    apply_progressive_prune_plan, load_pilot_checkpoint, save_pilot_checkpoint,
)


class ProgressiveResumeTests(unittest.TestCase):
    def test_resume_reconstructs_small_architecture_and_state(self):
        config = tiny_hnerv_config(fc_dim=12)
        model, image, embedding = tiny_hnerv(seed=4, fc_dim=12)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        optimizer.zero_grad()
        model(image)[0].square().mean().backward()
        optimizer.step()
        start_macs = measure_decoder_compute(model, embedding)['total_MACs']
        layers = get_channel_group_layers(model)
        keeps = [list(range(layer.channels)) for layer in layers]
        keeps[0].pop(0)
        plans, head = build_prune_plan_from_keep_sets(model, keeps)
        apply_progressive_prune_plan(model, optimizer, plans, head, verify_embedding=embedding)
        current_macs = measure_decoder_compute(model, embedding)['total_MACs']
        solver = PCDSolver(tau=0.07, beta=0.9, eps=1e-7)
        solver.t, solver.v = 5, [2.0, 3.0]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pilot.pth'
            save_pilot_checkpoint(
                path,
                original_config=config,
                model=model,
                optimizer=optimizer,
                method='m4_pcd', kappa=0.5,
                start_macs=start_macs, current_macs=current_macs,
                counted_training_seconds=12.5, budget_seconds=100,
                epoch=3, optimizer_steps=17,
                removal_history=[{'groups_removed': 1}],
                frozen_layer_costs={'decoder.1.conv.upconv.0': 1.0},
                solver=solver, metrics={'PSNR': 20.0},
            )
            resumed, resumed_optimizer, resumed_solver, payload = load_pilot_checkpoint(
                path, restore_rng=False
            )

        self.assertEqual(decoder_layer_widths(resumed), decoder_layer_widths(model))
        self.assertEqual(resumed.state_dict().keys(), model.state_dict().keys())
        self.assertTrue(all(
            torch.equal(value, resumed.state_dict()[name])
            for name, value in model.state_dict().items()
        ))
        self.assertEqual(payload['counted_training_seconds'], 12.5)
        self.assertEqual(payload['metrics']['PSNR'], 20.0)
        self.assertEqual(resumed_solver.t, 5)
        self.assertEqual(resumed_solver.v, [2.0, 3.0])
        self.assertTrue(resumed_optimizer.state)
        resumed_optimizer.zero_grad()
        resumed(image)[0].square().mean().backward()
        resumed_optimizer.step()


if __name__ == '__main__':
    unittest.main()
