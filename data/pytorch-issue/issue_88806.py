import torch.nn as nn

import unittest
import unittest.mock as mock

import torch
import torch.nn


class TestMultiheadAttentionFastPath(unittest.TestCase):

    @torch.no_grad()
    def test_multihead_attn_fast_path_eval_mode_with_dropout(self):
        """Ensure that fast path works with dropout, if we are in eval mode."""
        device = "cpu"
        embed_dim = 16
        num_heads = 8
        batch_size = 8
        src_len = 5

        query = value = key = torch.rand(batch_size, src_len, embed_dim).to(device)
        with mock.patch("torch._native_multi_head_attention") as fastpath_mock:
            mta_model = torch.nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True, device=device, dropout=0.1
            )
            # without eval mode and with dropout, the fast path should not be taken
            mta_model(query, key, value)
            self.assertFalse(fastpath_mock.called)

            # this should effectively disable dropout, making the check for dropout in fast_path irrelevant
            # and making `self.training` a falsy value, which should enable the fast_path
            mta_model.eval()
            mta_model(query, key, value)
            # This currently fails:
            self.assertTrue(fastpath_mock.called)


if __name__ == "__main__":
    unittest.main()