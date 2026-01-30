import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention

flex_attention = torch.compile(flex_attention, dynamic=False)

q = torch.randn((1, 1, 128, 16), dtype=torch.float16, device="cuda")
k = torch.randn((1, 1, 128, 16), dtype=torch.float16, device="cuda")
v = torch.randn((1, 1, 128, 16), dtype=torch.float16, device="cuda")
mass = torch.ones((1), dtype=torch.float16, device="cuda")

def score_mod(score, b, h, q_idx, kv_idx):
  return score + torch.log(mass[0])

out = flex_attention(q, k, v, score_mod=score_mod) # fails