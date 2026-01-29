# torch.rand(8, 10, 3, 2, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.prod(x, dim=3, keepdim=True)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with shape (8, 10, 3, 2) matching the first input in the repro
    return torch.rand(8, 10, 3, 2, dtype=torch.float32, requires_grad=True)

