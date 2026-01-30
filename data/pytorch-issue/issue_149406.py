import os
os.environ["TORCH_LOGS"] = "recompiles_verbose"
import torch
x = torch.randn((10, 10), device="cuda", requires_grad=False)

@torch.compile(dynamic=True)
def model(x, y):
    return x * y

y = model(x, 1.5)
y2 = model(x, 2.5)

import os
os.environ["TORCH_LOGS"] = "recompiles_verbose"

import torch
x = torch.randn((10, 10), device="cuda", requires_grad=False)

@torch.compile(dynamic=True)
def model(x, y):
    return x * y

y = model(x, 1)
y2 = model(x, 2)