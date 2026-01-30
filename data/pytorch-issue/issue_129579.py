import torch.nn as nn

import torch

b, h = 8, 2
s_q, s_kv = 128, 128
d_qk, d_v = 128, 144
q = torch.randn(b, h, s_q, d_qk, device='cuda:0', dtype=torch.bfloat16)
k = torch.randn(b, h, s_kv, d_qk, device='cuda:0', dtype=torch.bfloat16)
v = torch.randn(b, h, s_kv, d_v, device='cuda:0', dtype=torch.bfloat16)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v)