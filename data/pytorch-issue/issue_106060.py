import torch

class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.register_buffer("buffer", torch.rand([4], device="cuda"))

    def forward(self, x):
        # should be a no-op, but causes dynamo to lose the static input
        self.buffer = self.buffer.to(x)
        return self.buffer + x

@torch.compile(mode="reduce-overhead")
def foo(mod, inp):
    mod(inp)


m = Mod()
foo(m, torch.rand([4], device="cuda"))

import torch
from torch.func import functional_call
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch

class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

        self.buffer = torch.tensor([0.5, 0.5, 0.5, 0.5])

    def forward(self, x):
        # should be a no-op, but causes dynamo to lose the static input
        self.buffer = self.buffer.to(x)
        self.buffer.add_(x)
        return self.buffer + x

eager_mod = Mod()
eager_result = eager_mod(torch.tensor([0.5, 0.5, 0.5, 0.5], device="cuda"))
out_mod = torch._dynamo.optimize("eager", nopython=True)(Mod())
compiled_result = out_mod(torch.tensor([0.5, 0.5, 0.5, 0.5], device="cuda"))

setattr