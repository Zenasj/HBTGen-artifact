# torch.rand(5000, 5000, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x @ x  # Matrix multiplication as per original computation

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5000, 5000, dtype=torch.float32)

