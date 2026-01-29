# torch.rand(2, dtype=torch.float)  # Input is a tensor of shape (2,) with requires_grad=True
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch._foreach_sigmoid([x[0], x[1]])  # Split input into two 0D tensors and apply foreach_sigmoid

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float, requires_grad=True)  # Returns a tensor of shape (2,) with requires_grad enabled

