# torch.rand(3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x.copy_(3)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4)

