# Input: (torch.randn(4), torch.randn(8))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        x_view = x.view(-1, 4)
        y_view = y.view(-1, 4)
        return x_view * y_view

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(4), torch.randn(8))

