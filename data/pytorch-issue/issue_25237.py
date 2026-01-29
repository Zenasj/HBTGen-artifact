# torch.rand(()), torch.rand(())  # Input is a tuple of two 0-dimensional tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        if x > y:
            r = x
        else:
            r = x + y
        return r

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(()), torch.rand(()))

