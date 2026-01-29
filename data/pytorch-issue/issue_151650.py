# torch.rand(10), torch.rand(10)  # Two tensors of shape (10,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(10), torch.rand(10))

