# torch.rand(1, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        part1 = x[:, :1]
        part2 = x[:, -1:]
        return torch.stack((part1, part2), dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, dtype=torch.float32)

