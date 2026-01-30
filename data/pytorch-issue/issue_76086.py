import torch.nn as nn

import torch

class Base(torch.nn.Module):
    __constants__ = ["x"]
    x: float

    def __init__(self):
        super(Base, self).__init__()
        self.x = 5.

class Inside(torch.nn.Module):
    def __init__(self):
        super(Inside, self).__init__()

    def forward(self, x):
        return x + self.x

# if you replace Inside with Base, it works.
class Outside(torch.nn.Module):
    def __init__(self, mod: Inside):
        super(Outside, self).__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod.x + x

# If you replace Inside with Base, it works.
m = Outside(Inside())

x = torch.rand((2, 2))

m_s = torch.jit.script(m)
print(m_s(x) - x)