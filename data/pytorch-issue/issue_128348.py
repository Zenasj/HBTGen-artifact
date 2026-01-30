import torch.nn as nn

import torch

b, h, s, d_qk, d_v = 8, 2, 640, 128, 64
q = torch.randn(b, h, s, d_qk, device='cuda:0', dtype=torch.bfloat16)
k = torch.randn(b, h, s, d_qk, device='cuda:0', dtype=torch.bfloat16)
v = torch.randn(b, h, s, d_v, device='cuda:0', dtype=torch.bfloat16)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v)