# torch.rand(2, 3, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        b = a.t()
        b.mul_(1.0)
        return b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.bfloat16)

