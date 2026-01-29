# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        condition = torch.ones_like(x).to(torch.bool)
        x_val = torch.zeros_like(x)
        y_val = torch.ones_like(x) * 2
        return torch.where(condition, x_val, y_val)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)

