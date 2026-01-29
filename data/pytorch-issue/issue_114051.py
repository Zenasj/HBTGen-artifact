# torch.rand(2, 3, 8, 8, dtype=torch.float32)  # Input shape: B=2, C=3, H=8, W=8
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pinverse(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 8, 8, dtype=torch.float32)

