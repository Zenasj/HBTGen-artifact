import torch
from torch import nn

# torch.rand(3, 3, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, x):
        return torch.sin(x) + torch.cos(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)

