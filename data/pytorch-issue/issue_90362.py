# torch.rand(4, dtype=torch.float32), torch.rand(4, dtype=torch.float32)  # Input is a tuple of two tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = -b
        return x * b

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(4, dtype=torch.float32)
    b = torch.rand(4, dtype=torch.float32)
    return (a, b)

