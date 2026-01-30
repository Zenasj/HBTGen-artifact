import torch.nn as nn

import torch

def make_mod():
    x = torch.randn(10, 10)
    class Mod(torch.nn.Module):
        @property
        def y(self):
            return torch.randn(10, 10) + x

        def forward(self, x):
            return x @ self.y

    return Mod

mod = make_mod()()

def fn(x):
    return mod(x)

opt_fn = torch.compile(fn, backend="eager")

inp = torch.randn(10, 10)
opt_fn(inp)