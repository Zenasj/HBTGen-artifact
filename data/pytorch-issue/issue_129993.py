import torch.nn as nn

import torch

class Mod(torch.nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.mod, name)

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, *args):
        return self.mod(*args)

mod = torch.nn.Linear(3, 3)

def fn(mod, x):
    mod = Mod(mod)
    if hasattr(mod, "asdf"):
        return x + 1
    return x + 2

torch.compile(fn, backend="eager")(mod, torch.randn(3, 3))