import torch.nn as nn

import torch
from torch import nn

class Wrapper():
    def __init__(self, t):
        self.t = t
    def to(self, device: "Device" = None):  # noqa
        return self.t.to(device=device)
class A(nn.Module):
    def forward(self):
        return Wrapper(torch.rand(4, 4))

scripted = torch.jit.script(A())
scripted.save("tmp.pt")

import torch
from torch import nn

class Wrapper():
    def __init__(self, t):
        self.t = t
    def to(self, device: torch.device = None):
        return self.t.to(device=device)
class A(nn.Module):
    def forward(self):
        return Wrapper(torch.rand(4, 4))

scripted = torch.jit.script(A())
scripted.save("tmp.pt")