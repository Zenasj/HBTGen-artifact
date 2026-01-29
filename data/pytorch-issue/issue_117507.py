# Input is a tuple of (a, b) where a.shape=(1, 3, 2) and b.shape=(1, 3, 6250000)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x
        return torch.linalg.lstsq(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(1, 3, 2, device='cuda')
    b = torch.rand(1, 3, 2500 * 2500, device='cuda')
    return (a, b)

