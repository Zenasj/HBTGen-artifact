# torch.rand(2, 2, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        b = x.view(2, 2)
        return b.sgn()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32, requires_grad=True)

