import torch.nn as nn

import torch

scale = 0.125

# 5 tokens, 1 padding
seq_length = 5
max_seq_length = 10

q = torch.randn(1, 2, seq_length + 1, 8)
k = torch.randn(1, 2, max_seq_length, 8)
v = torch.randn(1, 2, max_seq_length, 8)
mask = torch.tensor([[[
    [ True, False, False, False, False, False, False, False, False, False],  # regular mask
    [ True,  True, False, False, False, False, False, False, False, False],
    [ True,  True,  True, False, False, False, False, False, False, False],
    [ True,  True,  True,  True, False, False, False, False, False, False],
    [ True,  True,  True,  True,  True, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False]]]])  # padding mask



def torch_sdpa(q, k, v, mask):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.0, scale=scale)


def naive_sdpa_1(q, k, v, mask):
    att = (q @ k.transpose(-2, -1)) * scale
    att = torch.masked_fill(att, ~mask, float("-inf"))
    att = torch.nn.functional.softmax(att, dim=-1)
    return att @ v


def naive_sdpa_2(q, k, v, mask):
    att = (q @ k.transpose(-2, -1)) * scale
    att = torch.masked_fill(att, ~mask, torch.finfo(att.dtype).min)
    att = torch.nn.functional.softmax(att, dim=-1)
    return att @ v

y = torch_sdpa(q, k, v, mask)
print(torch.isnan(y).any())

y = naive_sdpa_1(q, k, v, mask)
print(torch.isnan(y).any())

y = naive_sdpa_2(q, k, v, mask)
print(torch.isnan(y).any())

import torch
import torch.nn.functional as F

q = torch.eye(32, 32, device='cuda').expand(1, 1, 32, 32).contiguous()
k = torch.eye(32, 32, device='cuda').expand(1, 1, 32, 32).contiguous()
v = torch.eye(32, 32, device='cuda').expand(1, 1, 32, 32).contiguous()

attn_mask = torch.ones(1, 1, 32, 32, device='cuda').to(torch.bool)
attn_mask[..., :16, 16:] = 0
res1 = F.scaled_dot_product_attention(q, k, v, attn_mask)
res2 = F.scaled_dot_product_attention(q, k, v, attn_mask.to(torch.float))

print(res1)
print(res2)

import torch
import math
import torch.nn.functional as F

def f(attn_mask):
    res = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
    return res @ v


k = torch.eye(32, 32, device='cuda').expand(1, 1, 32, 32).contiguous()
v = torch.eye(32, 32, device='cuda').expand(1, 1, 32, 32).contiguous()
q = torch.eye(32, 32, device='cuda').expand(1, 1, 32, 32).contiguous()

attn_mask = torch.ones(1, 1, 32, 32, device='cuda').to(torch.bool)
attn_mask[..., :16, 16:] = 0

res1 = f(attn_mask.to(torch.float))
res2 = F.scaled_dot_product_attention(q, k, v, attn_mask.to(torch.float))

print((res1 - res2).abs().max())