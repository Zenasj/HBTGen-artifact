# torch.rand(1, 1, 0, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, z = inputs
        return torch.einsum('abc,a,b->c', x, y, z)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1, 1, 0)
    y = torch.rand(1)
    z = torch.rand(1)
    return (x, y, z)

