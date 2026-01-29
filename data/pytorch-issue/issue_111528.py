# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        @torch.no_grad
        def inner(x):
            return x.sin()
        return inner(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32)

