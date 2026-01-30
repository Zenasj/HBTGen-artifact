import torch.nn as nn

import torch

torch.manual_seed(42)
device = "cuda:0"
dtype = torch.float32

B = 4
H = 12
N = 2**12
D = 1024

q = torch.rand(B, H, N, D, dtype=dtype, device=device)
k = torch.rand(B, H, N, D, dtype=dtype, device=device)
v = torch.rand(B, H, N, D, dtype=dtype, device=device)

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

import torch

torch.manual_seed(42)
device = "cuda:0"
dtype = torch.float32

B = 4
H = 12
N = 2**12
D = 1024

q = torch.rand(B, H, N, D, dtype=dtype, device=device)
k = torch.rand(B, H, N, D, dtype=dtype, device=device)
v = torch.rand(B, H, N, D, dtype=dtype, device=device)

m = torch.ones((N, N), dtype=torch.bool, device=device).triu(N - N + 1)
m = m.float().masked_fill(m, float("-inf"))

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m)

import torch
import xformers.ops as xops

torch.manual_seed(42)
device = "cuda:0"
dtype = torch.float32

B = 4
H = 12
N = 2**12
D = 1024

q = torch.rand(B, H, N, D, dtype=dtype, device=device)
k = torch.rand(B, H, N, D, dtype=dtype, device=device)
v = torch.rand(B, H, N, D, dtype=dtype, device=device)

m = torch.ones((N, N), dtype=torch.bool, device=device).triu(N - N + 1)
m = m.float().masked_fill(m, float("-inf"))

# xformers attention expects shape B, N, H, D instead of B, H, N, D
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

out = xops.memory_efficient_attention(q, k, v, attn_bias=m)