# torch.rand(B, 1, 2, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + 2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 2, 1, dtype=torch.float32)

