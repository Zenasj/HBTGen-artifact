# (torch.rand(10), torch.rand(10))  # Tuple of two tensors each of shape (10,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = b * -1
        return x * b

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(10)
    b = torch.rand(10)
    return (a, b)

