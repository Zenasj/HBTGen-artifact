import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention


flex_attention = torch.compile(flex_attention)


x = torch.randn(
    (1, 8, 256, 128),
    device='cuda',
    dtype=torch.float,
    requires_grad=True
)

flex_attention(x, x, x).sum().backward()

import torch
from torch.nn.attention.flex_attention import flex_attention

flex_attention = torch.compile(flex_attention)

x = torch.randn(
    (1, 8, 256, 128),
    device='cuda',
    dtype=torch.float,
    requires_grad=True
)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
    output = flex_attention(x, x, x)

output.sum().backward()

import torch
from torch.nn.attention.flex_attention import flex_attention

flex_attention = torch.compile(flex_attention)

emb = torch.nn.Embedding(128, 128, dtype=torch.float, device='cuda')
lin = torch.nn.Linear(128, 128, dtype=torch.float, device='cuda', bias=False)

x = torch.randint(
    0, 128,
    size=(1, 8, 256),
    device='cuda',
)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
    x = emb(x)
    x = flex_attention(x, x, x)
    x = lin(x)
    # x = flex_attention(x, x, x)  # uncomment this to break the code

x.sum().backward()