# (torch.rand(1), torch.rand(3), torch.rand(3))  # input tuple (x, y, z)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, z = inputs
        def f():
            return y + 2
        def g():
            return z + 1
        return torch.cond(x, f, g, ())  # Fourth argument is empty tuple to avoid error

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1), torch.rand(3), torch.rand(3))

