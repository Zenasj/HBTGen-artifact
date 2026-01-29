import torch
import operator
from torch import nn

# (torch.rand(4), torch.rand(4))  # Input is a tuple of two tensors of shape (4,)
class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x * operator.pos(y)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(4), torch.rand(4))

