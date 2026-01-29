# torch.rand(10, 10, dtype=torch.float32, device='cuda:0')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 10, device='cuda:0')

