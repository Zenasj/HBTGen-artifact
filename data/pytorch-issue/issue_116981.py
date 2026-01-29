# torch.rand(2, 2, dtype=torch.float), torch.rand(2, 2, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        z = x * y
        for _ in range(10):
            z = z * y
        return z

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2, 2), torch.rand(2, 2))

