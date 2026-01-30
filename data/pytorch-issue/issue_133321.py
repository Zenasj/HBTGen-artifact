import torch.nn as nn

Python
from functools import lru_cache
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import torch


torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

B = 2
H = 16
S = 28800
D = 96

query = torch.randn(
    B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
)
key = torch.randn(
    B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
)
value = torch.randn(
    B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
)

d_k, seq_len = query.size(-1), query.size(-2)
block_size = seq_len // 8
frame_size = seq_len // 2

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask

def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
    row_mask = q_idx % frame_size < block_size
    col_mask = kv_idx % frame_size < block_size
    diagonal_mask = (q_idx // block_size) == (kv_idx // block_size)
    return row_mask | col_mask | diagonal_mask

def noop(b, h, q_idx, kv_idx):
    return True

block_mask = create_block_mask_cached(prefix_lm_causal_mask, 1, 1, seq_len, seq_len)
hidden_states = flex_attention(query, key, value, block_mask=block_mask)