# torch.rand(2, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.arctanh(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 100, dtype=torch.float32)

