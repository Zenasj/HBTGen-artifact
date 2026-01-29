# torch.rand(100, dtype=torch.int)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.sum() // 5

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 100, (100,), dtype=torch.int)

