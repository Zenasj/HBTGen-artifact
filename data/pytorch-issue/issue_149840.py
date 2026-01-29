# torch.rand(1, 128, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        for _ in range(8):
            x = x * 3
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 128, dtype=torch.bfloat16)

