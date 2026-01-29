# torch.rand(10000, 10000, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.bernoulli(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros(10000, 10000, dtype=torch.float32)

