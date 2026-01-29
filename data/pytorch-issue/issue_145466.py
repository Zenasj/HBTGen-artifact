# torch.rand(32, 32, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args):
        cos = torch.cos(args[0])
        floor = torch.floor(cos)
        return floor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 32, dtype=torch.float16)

