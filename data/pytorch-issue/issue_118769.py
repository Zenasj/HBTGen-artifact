# torch.rand(4, 8), torch.rand(8, 32)  # Input shapes (x, y)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.mm(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(4, 8, requires_grad=True)
    y = torch.randn(8, 32, requires_grad=True)
    return (x, y)

