# torch.rand(3, device='cuda'), torch.rand(3)  # Input shapes for x and y (as a tuple)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x + y.sum()

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(3, device='cuda')
    y = torch.randn(3)
    return (x, y)

