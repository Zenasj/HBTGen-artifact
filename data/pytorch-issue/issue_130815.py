# torch.rand(10, 10), torch.rand(10, 10)  # Two input tensors of shape (10,10)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(10, 10), torch.randn(10, 10))

