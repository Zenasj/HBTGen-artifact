# torch.rand(4, dtype=torch.complex128, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.mean(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.complex128, device='cuda')

