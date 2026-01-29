# torch.rand(B, C, L, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=0.5, mode="linear")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 5, 65536, dtype=torch.float32)

