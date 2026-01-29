# (torch.tensor([False, True], dtype=torch.bool), torch.ones(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float64))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        condition, x, y = inputs
        return torch.where(condition, x, y)

def my_model_function():
    return MyModel()

def GetInput():
    condition = torch.tensor([False, True], dtype=torch.bool)
    x = torch.ones(2, dtype=torch.float32)
    y = torch.zeros(2, dtype=torch.float64)
    return (condition, x, y)

