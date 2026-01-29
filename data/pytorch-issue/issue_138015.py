# torch.rand(1, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.randn(1, dtype=torch.bfloat16)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.bfloat16)

