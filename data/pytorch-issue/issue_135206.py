import torch.nn as nn
import random

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import numpy as np

B, H, S, D = 4, 16, 2048, 128
data_dtype = torch.bfloat16
paddings_tensor = torch.zeros((B), device="cuda")

def truncate_block_mask(b, h, q_idx, kv_idx):
    return torch.where(q_idx < paddings_tensor[b], torch.where(kv_idx < paddings_tensor[b], True, False), False)

def flex_attn_wrapper(q, k, v):
    return flex_attention(q, k, v)

def flex_attn_block_mask_wrapper(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

flex_attn_compiled = torch.compile(flex_attn_wrapper, dynamic=True)
flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=True)

q = torch.randn(B, H, S, D, device="cuda", dtype=data_dtype, requires_grad=True)
k = torch.randn(B, H, S, D, device="cuda", dtype=data_dtype, requires_grad=True)
v = torch.randn(B, H, S, D, device="cuda", dtype=data_dtype, requires_grad=True)

for _ in range(10):        
    paddings_tensor[:B] = torch.tensor(np.random.randint(0.7*S, S, size=B), device="cuda")
    block_mask = create_block_mask(truncate_block_mask, B, H=None, Q_LEN=S, KV_LEN=S, device="cuda", _compile=True)

    # this works
    out_vanilla = flex_attn_compiled(q, k, v)
    # this fails
    out_masked = flex_attn_block_mask_compiled(q, k, v, block_mask)