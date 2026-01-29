# torch.rand(2, 4, 3, 4), torch.rand(2, 4, 4, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.matmul(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2, 4, 3, 4)
    y = torch.rand(2, 4, 4, 3)
    return (x, y)

