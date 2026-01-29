# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = torch.empty_like(x)
        torch.clamp(x, min=0, out=y)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 3, 3, dtype=torch.float32)

