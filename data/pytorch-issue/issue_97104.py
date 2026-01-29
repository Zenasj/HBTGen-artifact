# Input: (torch.rand(8, 2), torch.rand(8) > 0.5, torch.rand([]))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, z = inputs
        return torch.index_put(x, [y], z)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(8, 2)
    y = torch.rand(8) > 0.5
    z = torch.rand([])
    return (x, y, z)

