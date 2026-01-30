import torch
import os

def foo(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.cos(x)

print("main pid ", os.getpid())
x = torch.rand(3, 3)
x_eager = foo(x)
x_pt2 = torch.compile(foo)(x)