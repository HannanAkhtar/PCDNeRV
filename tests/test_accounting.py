"""Parameter-accounting consistency tests (deployment-team issue #9)."""

import unittest

from helpers import tiny_model
from shared.accounting import param_accounting, decoder_head_state, count_params
from train_pcd_nerv import make_tag


class TestParamAccounting(unittest.TestCase):
    def test_scopes_partition_trainable_params(self):
        model = tiny_model()
        acct = param_accounting(model, embedding_storage=1234)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(
            acct['encoder_params'] + acct['decoder_head_params'] + acct['other_params'],
            acct['trainable_params'])
        self.assertEqual(acct['trainable_params'], total)
        # NeRV-PE: no encoder
        self.assertEqual(acct['encoder_params'], 0)
        self.assertEqual(acct['embedding_storage'], 1234)
        self.assertEqual(
            acct['total_stored_representation'],
            acct['decoder_head_params'] + 1234)

    def test_decoder_head_state_matches_scope_count(self):
        model = tiny_model()
        acct = param_accounting(model)
        self.assertEqual(count_params(decoder_head_state(model)),
                         acct['decoder_head_params'])


class TestRunTags(unittest.TestCase):
    def test_tags_are_method_and_seed_safe(self):
        tags = {
            make_tag('baseline', 'conv', 0.05, 1e-3, 1e-5, seed=1),
            make_tag('prox_gl', 'conv', 0.05, 1e-3, 1e-5, seed=1),
            make_tag('weighted_sum', 'conv', 0.05, 1e-3, 1e-5, seed=1),
            make_tag('pcd', 'conv', 0.05, 1e-3, 1e-5, seed=1),
            make_tag('pcd', 'conv', 0.05, 1e-3, 1e-5, seed=2),
            make_tag('pcd', 'conv', 0.05, 1e-3, 1e-5, seed=1, finetune=True),
        }
        self.assertEqual(len(tags), 6)  # all distinct
        self.assertEqual(make_tag('pcd', 'conv', 0.01, 1e-3, 0, seed=1),
                         'pcd_conv_tau0.01_lprox1e-03_seed1')
        self.assertEqual(make_tag('baseline', 'conv', None, None, None, seed=3),
                         'baseline_seed3')
        for t in tags:
            self.assertIn('seed', t)


if __name__ == '__main__':
    unittest.main()
