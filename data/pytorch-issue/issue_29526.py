# torch.rand(1, 2, dtype=torch.float32)
import torch
from torch import nn

class Dummy(nn.Module):
    def forward(self, x):
        return x

class MyModel(nn.Module):
    def __init__(self, dummies: nn.ModuleList):
        super().__init__()
        self._dummies = dummies
        
    def forward(self, x):
        out = []
        for dummy in self._dummies:
            out.append(dummy(x))
        return tuple(out)

def my_model_function():
    # Trace Dummy module and initialize MyModel with ModuleList
    dummy = torch.jit.trace(Dummy(), torch.randn(1, 2))
    dummies = nn.ModuleList([dummy])
    return MyModel(dummies)

def GetInput():
    # Generate input tensor matching traced module's example input
    return torch.randn(1, 2)

