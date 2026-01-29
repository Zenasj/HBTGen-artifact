# torch.rand(4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.select(x, x.dim()-1, 0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4)

