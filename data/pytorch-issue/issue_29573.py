from typing import NamedTuple, Optional
import torch.nn as nn
import torch

class S(NamedTuple):
    s: Optional[int]

class B(nn.Module):
    def forward(self):
        return S(None)

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = B()
        self.x = torch.zeros(20)
    def forward(self):
        s = self.mod.forward()
        if s.s is None:
            return self.x[0]
        else:
            assert s.s is not None
            return self.x[s.s]

torch.jit.script(A())

start = 0 if s.start is None else s.start                                
assert start is not None  # why is this necessary?