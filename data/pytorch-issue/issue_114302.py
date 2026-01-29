# torch.rand(6, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = x.expand(2, *x.shape)
        y[0, 0] = 5
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(6, dtype=torch.float32)

