# torch.randn(32, 256, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.linalg.matrix_power(x, 0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 256, 1, 1, dtype=torch.float32)

