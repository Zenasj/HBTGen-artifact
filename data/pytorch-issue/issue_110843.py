# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, flat_p):
        p = flat_p[0:2]
        with torch.no_grad():
            p.set_(flat_p[0:2])
        return flat_p

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, requires_grad=True)

