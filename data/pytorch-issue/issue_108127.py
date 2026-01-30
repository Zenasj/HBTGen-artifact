import torch

x = torch.randn(2, 100, 256, device="cuda", dtype=torch.float16)
y = torch.randn(2, 100, 256, device="cuda")

with torch.autocast("cuda", torch.float16):
    out = torch.linalg.vecdot(x, y)
    # out = (x.unsqueeze(-2) @ y.unsqueeze(-1)).squeeze(-1)  # this works
    # out = (x * y).sum(-1)  # this works, but promotes x to float32 and calculates in float32
    # out = torch.einsum("...i,...i->...", x, y)  # this works too

print(out.dtype)

import torch
import numpy as np


@torch.compile()
def fn(x, y):
    return torch.dot(x, y)


x = torch.randn(512, device="cuda")
y = torch.randn(512, device="cuda")
fn(x,y)