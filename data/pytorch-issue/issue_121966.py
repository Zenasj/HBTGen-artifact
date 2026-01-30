import torch.nn as nn

import time
import torch
import torch.utils.checkpoint

def add_and_drop(x):
    return torch.nn.functional.dropout(x * 5, 0.5)

x = torch.rand((50001, 3072), device='cuda', dtype=torch.bfloat16)

x.requires_grad_(True)
x.retain_grad()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    f = torch.compile(add_and_drop)
    out = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
    out.backward(torch.rand_like(out))
    print(x.grad)