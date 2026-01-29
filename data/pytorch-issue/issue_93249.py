# torch.rand(B, 8, 2, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.min(1).values

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 8, 2, dtype=torch.float16)

