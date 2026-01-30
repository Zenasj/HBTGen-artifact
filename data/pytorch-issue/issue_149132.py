import torch.nn as nn

import torch
import torch.nn.functional as F

print(torch.__version__)

device = torch.device('mps')

B=2
T=3
n_kv_head = 2
n_q_head = 4
dim = 8

attn_mask = torch.ones((T, T)).to(device)


q = torch.rand(B, n_q_head, T, dim).to(device)
k = torch.rand(B, n_kv_head, T, dim).to(device)
v = torch.rand(B, n_kv_head, T, dim).to(device)

F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)