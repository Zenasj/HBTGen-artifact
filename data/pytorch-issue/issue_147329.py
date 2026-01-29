# torch.rand(4, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.isin(x, 1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 5, (4,), dtype=torch.int64)

