# torch.rand(2, 3, 4, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.empty_like(x).exponential_()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

