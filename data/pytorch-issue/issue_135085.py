import torch.nn as nn

import torch

device='xpu'
Z=1
H=32
N_CTX=16384 # reduce helps
D_HEAD=64
causal = False
dtype = torch.float16
sm_scale = 0.125

q = torch.randn((Z, H, N_CTX, D_HEAD), device=device, dtype=dtype)
k = torch.randn((Z, H, N_CTX, D_HEAD), device=device, dtype=dtype)
v = torch.randn((Z, H, N_CTX, D_HEAD), device=device, dtype=dtype)


torch.nn.functional.scaled_dot_product_attention(
           q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=sm_scale)