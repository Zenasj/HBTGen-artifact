# Two tensors of shape (2, 3, 4), dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.pow(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2, 3, 4, dtype=torch.float32), torch.rand(2, 3, 4, dtype=torch.float32))

