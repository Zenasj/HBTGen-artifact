# torch.rand(B, C, H, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.split(1, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 384, 2, dtype=torch.float32)

