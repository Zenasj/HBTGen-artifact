# torch.rand(10, 3), torch.rand(11, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return torch.cdist(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randn(10, 3)
    b = torch.randn(11, 3)
    return (a, b)

