# torch.rand(B, 0, 0, dtype=torch.float32)  # Empty 0x0 matrix batch input
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return torch.matrix_exp(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Matches the PR's test case (batch size 1)
    return torch.rand(B, 0, 0, dtype=torch.float32)  # Empty 0x0 matrix tensor

