# Input: (torch.randint(-10, 10, (10, 10), dtype=torch.int64), torch.randint(-10, 10, (10,), dtype=torch.int64))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.diagonal_scatter(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randint(-10, 10, (10, 10), dtype=torch.int64)
    y = torch.randint(-10, 10, (10,), dtype=torch.int64)
    return (x, y)

