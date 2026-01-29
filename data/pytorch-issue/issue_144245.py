# torch.rand(B, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, size=3, mode='linear', align_corners=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, dtype=torch.float32)

