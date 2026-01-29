# torch.rand(1, 1, 192, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.expand(x.shape[0], -1, -1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 192, dtype=torch.float32)

