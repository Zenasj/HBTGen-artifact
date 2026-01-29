# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x, y=3):
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

