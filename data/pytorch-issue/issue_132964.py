# torch.rand(512, 1048576, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(dim=self.dim)

def my_model_function():
    return MyModel(dim=-1)

def GetInput():
    return torch.rand(512, 1048576, dtype=torch.float32, device='cuda')

