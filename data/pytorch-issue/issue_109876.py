# torch.rand(5,5, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = torch.mean(x)
        z = torch.sum(x)
        return y, z  # Returns tuple of outputs to enable comparison of their backward logic

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 5, dtype=torch.float32, requires_grad=True)

