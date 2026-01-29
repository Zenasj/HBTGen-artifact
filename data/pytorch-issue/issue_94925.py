# torch.rand((), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.roll(x, 1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(0.0)

