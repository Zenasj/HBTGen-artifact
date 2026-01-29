# torch.rand(B, C, H, W, dtype=torch.float32)  # Example input shape: (4,4,1,1)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.sin().cos()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, 1, 1, dtype=torch.float32)

