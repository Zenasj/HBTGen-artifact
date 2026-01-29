# torch.rand(B, 3, dtype=torch.bfloat16)  # Each tensor in the input tuple has this shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.cross(x, y, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(100, 3, dtype=torch.bfloat16)
    y = torch.rand(100, 3, dtype=torch.bfloat16)
    return (x, y)

