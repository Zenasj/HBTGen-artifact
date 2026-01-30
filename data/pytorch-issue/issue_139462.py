import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch.manual_seed(1234)

def mask_mod(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    return causal_mask

mask_1 = create_block_mask(
    mask_mod = mask_mod,
    B = 2,
    H = None,
    Q_LEN = 128,
    KV_LEN = 128,
    device = "cuda",
)

mask_2 = create_block_mask(
    mask_mod = mask_mod,
    B = 2,
    H = None,
    Q_LEN = 128,
    KV_LEN = 256,
    device = "cuda",
)

flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

shape = (2, 1, 2, 16)
q = torch.normal(0.0, 3.0, shape, device = "cuda")
k = torch.normal(0.0, 3.0, shape, device = "cuda")
v = torch.normal(0.0, 3.0, shape, device = "cuda")

y0 = F.scaled_dot_product_attention(q, k, v, is_causal = True)

y1 = flex_attention(q, k, v, block_mask = mask_1)
y2 = flex_attention(q, k, v, block_mask = mask_2)

y3 = flex_attention_compiled(q, k, v, block_mask = mask_1)
y4 = flex_attention_compiled(q, k, v, block_mask = mask_2)

print(y0.sum())
print(y1.sum())
print(y2.sum())
print(y3.sum())
print(y4.sum())

print(y0[1])
print(y4[1])