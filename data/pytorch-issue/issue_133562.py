import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
from torch.nn.attention.flex_attention import _convert_mask_to_block_mask, _create_sparse_block_from_block_mask

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

B = 2
H = 16
S = 8192
D = 64

query = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
key = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
value = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
attn_mask = torch.randint(0, 2, (B, S, S), device="cuda", dtype=torch.bool) >= 1

BLOCK_SIZE = 2
block_mask = _create_sparse_block_from_block_mask(
    _convert_mask_to_block_mask(attn_mask, BLOCK_SIZE, BLOCK_SIZE),
    None,
    KV_BLOCK_SIZE=BLOCK_SIZE,
    Q_BLOCK_SIZE=BLOCK_SIZE,
)
a = flex_attention(query, key, value, block_mask=block_mask)