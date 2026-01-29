# Input is a tuple of two tensors each of shape (1, 1, 1, 5), dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x
        return a * b

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(1, 1, 1, 5, dtype=torch.float32, requires_grad=True)
    b = torch.rand(1, 1, 1, 5, dtype=torch.float32, requires_grad=True)
    return (a, b)

