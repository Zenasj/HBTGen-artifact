# torch.rand(5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x0):
        return torch.clamp(x0, max=6)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 5)

