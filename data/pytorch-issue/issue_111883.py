# torch.rand(10), torch.rand(10)  # Input is a tuple of two tensors of shape (10,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        a, b = inputs
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = -b
        return x * b

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(10), torch.randn(10))

