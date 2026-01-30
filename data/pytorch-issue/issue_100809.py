import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):
    def forward(self, x, y):
        x += 0.1 * y
        return x

x = torch.randn(5, 1)
y = torch.randn(5, 20)

func = Model()
jit_func = torch.compile(func)

res2 = jit_func(x.clone(), y)
print(res2.shape) # torch.Size([5, 20])

res1 = func(x.clone(), y) # without jit
# RuntimeError: output with shape [5, 1] doesn't match the broadcast shape [5, 20]

import torch
from functorch import functionalize

def f(x, y):
    x_ = x + 1
    x_ += 0.1 * y
    return x_



x = torch.ones(5, 1)
y = torch.ones(5, 20)

res2 = functionalize(f)(x, y)
print(res2.shape) # torch.Size([5, 20])