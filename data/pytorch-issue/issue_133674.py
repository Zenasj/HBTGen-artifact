import torch.nn as nn

import torch

from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask

B = 1
H = 8
S = 256
D = 128
D_L = 512

query = torch.randn(
    B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=False
)
key = torch.randn(
    B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=False
)
value = torch.randn(
    B, H, S, D_L, device="cuda", dtype=torch.bfloat16, requires_grad=False
)

def mask_fn(b, h, q_idx, kv_idx):
    return (
        q_idx == kv_idx
    )

def noop(b, h, q_idx, kv_idx):
    return True

block_mask = create_block_mask(mask_fn, B=B, H=H, Q_LEN=S, KV_LEN=S)
attn_out = flex_attention(query, key, value, block_mask=block_mask)
print("Attn_out shape without compile", attn_out.shape)

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

block_mask = create_block_mask(mask_fn, B=B, H=H, Q_LEN=S, KV_LEN=S)
attn_out = flex_attention(query, key, value, block_mask=block_mask)
print("Attn_out shape with compile", attn_out.shape)