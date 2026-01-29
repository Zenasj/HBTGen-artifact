# torch.rand(1, 1, 1, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x + 1
        x = x + 1
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 10, dtype=torch.float32)

