import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class TestModule(torch.nn.Module):
    def forward(self, q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

q = torch.randn(1, 1, 2048, 128, device="cuda:1", dtype=torch.bfloat16)
k = torch.randn(1, 1, 2048, 128, device="cuda:1", dtype=torch.bfloat16)
v = torch.randn(1, 1, 2048, 128, device="cuda:1", dtype=torch.bfloat16)
mask = create_block_mask(
    lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
    B=None,
    H=None,
    Q_LEN=2048,
    KV_LEN=2048,
    device="cuda:1",
)
mod = torch.compile(TestModule())
attn_output = mod(q, k, v, mask)
print(attn_output.shape)