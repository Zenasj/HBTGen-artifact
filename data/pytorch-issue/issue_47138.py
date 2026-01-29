# torch.rand(16, 64, 16, 16, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.fixup_bias2a = nn.Parameter(torch.zeros(1))
        self.fixup_scale = nn.Parameter(torch.ones(1))
        self.fixup_bias2b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        out = x
        out = out + self.fixup_bias2a
        out = out * self.fixup_scale + self.fixup_bias2b
        return out * out + identity

def my_model_function():
    return MyModel(64, 64)

def GetInput():
    return torch.randn(16, 64, 16, 16, dtype=torch.float16)

