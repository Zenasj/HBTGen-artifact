# torch.rand(3, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.sin()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32, device='cuda')

