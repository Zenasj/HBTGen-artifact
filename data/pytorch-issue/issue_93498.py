import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

import time

torch.set_num_interop_threads(1)
torch.set_num_threads(1) 

def mish(x: torch.Tensor):
    return x * torch.tanh(F.softplus(x))

ops = [
    F.mish,
    mish,
    torch.jit.script(mish),
    # torch.compile(mish, fullgraph=True),
    torch.compile(mish, mode="max-autotune", fullgraph=True),
    # torch.compile(mish, mode="reduce-overhead", fullgraph=True),
]

t = torch.randn(10000000)

torch._inductor.config.debug = True

# Warm up
for _ in range(10):
    for op in ops:
        _ = op(t)

res = {}
N = 10
for i, op in enumerate(ops):
    start = time.time_ns()
    for _ in range(N):
        _ = op(t)
    end = time.time_ns()
    
    res[f"{i}_{op}"] = f"{(end - start) / 1e6 / N:.2f} ms"

print(res)