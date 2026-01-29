# torch.rand((), dtype=torch.float16), torch.rand(0, dtype=torch.float16)  # Input is a tuple of two tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.true_divide(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand((), dtype=torch.float16)
    y = torch.rand(0, dtype=torch.float16)
    return (x, y)

