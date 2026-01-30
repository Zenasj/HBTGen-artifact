import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from triton.testing import do_bench

torch.set_default_device("cuda")
flex_attention = torch.compile(flex_attention, dynamic=True)

def create_block_causal_mask(sequence_ids):
    def block_causal_mask_fn(b, h, q_idx, kv_idx):
        return sequence_ids[b, q_idx] >= sequence_ids[b, kv_idx]
    B, seqlen = sequence_ids.shape
    return create_block_mask(block_causal_mask_fn, B, 1, seqlen, seqlen)


q = torch.randn(8, 8, 8192, 64, dtype=torch.float16)
k = torch.randn(8, 8, 8192, 64, dtype=torch.float16)
v = torch.randn(8, 8, 8192, 64, dtype=torch.float16)

sequence_ids = torch.cat(
    [torch.arange(375 + i).repeat(375 + i, 1).transpose(-1, -2).reshape(-1)[:8192][None, :] for i in range(8)],
    dim=0,
)

block_causal_mask = create_block_causal_mask(sequence_ids)

print("Sparsity: ", block_causal_mask.sparsity())
print("Flex (w/o kernel options): ", do_bench(lambda: flex_attention(q, k, v, block_mask=block_causal_mask)))
print("Flex (w kernel options): ", do_bench(lambda: flex_attention(q, k, v, block_mask=block_causal_mask, kernel_options={'BLOCK_M':64, 'BLOCK_N':64})))