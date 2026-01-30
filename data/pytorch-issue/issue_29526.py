import torch.nn as nn

import torch
from torch import nn

class Dummy(nn.Module):
    
    def forward(self, x):
        return x

class Model(nn.Module):

    def __init__(self, dummies: nn.ModuleList):
        super().__init__()
        self._dummies = dummies
        
    def forward(self, x):
        out = []
        for dummy in self._dummies:
            out.append(dummy(x))
        return tuple(out)
    
dummy = torch.jit.trace(Dummy(), torch.randn(1, 2))
dummies = nn.ModuleList([dummy])
model = Model(dummies)
script = torch.jit.script(model)