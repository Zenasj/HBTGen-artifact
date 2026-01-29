# torch.rand(100, 200, 300, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=-2, end_dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 200, 300)

