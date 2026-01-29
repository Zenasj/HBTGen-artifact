# torch.rand(50000, 1, dtype=torch.float32, requires_grad=True, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.pdist(x, p=2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(50000, 1, dtype=torch.float32, requires_grad=True, device='cuda')

