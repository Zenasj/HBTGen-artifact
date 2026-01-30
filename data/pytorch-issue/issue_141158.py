py
import torch
import dataclasses

def f(x):
    if dataclasses.is_dataclass(x):
        return x
    return x.sin()

x = torch.randn(3)
torch.compile(f, fullgraph=True)(x)