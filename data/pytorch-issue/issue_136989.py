import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention = torch.compile(flex_attention)
attn_mask = torch.ones((4,1,2048,2048), dtype=torch.bool, device='cuda').tril()

def causal(b, h, q_idx, kv_idx):
    h_ = h.new_zeros(h.shape)
    # print(b)  # uncomment this line to make the code work
    return attn_mask[b][h_][q_idx][kv_idx]
block_mask = create_block_mask(causal, B=4, H=None, Q_LEN=2048, KV_LEN=2048)
print(block_mask)


q, k, v = torch.randn(4, 1, 2048, 64, device='cuda'), torch.randn(4, 1, 2048, 64, device='cuda'), torch.randn(4, 1,2048, 64, device='cuda')

print(flex_attention(q, k, v, block_mask=block_mask))