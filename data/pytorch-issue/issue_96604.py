# torch.rand((), dtype=torch.uint8, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = torch.neg(x)
        return x < y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(5, device='cuda', dtype=torch.uint8)

