# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Applies reflection padding to last 3 dimensions (D, H, W) with padding [1,1] on D
        return F.pad(x, [0, 0, 0, 0, 1, 1], mode='reflect')

def my_model_function():
    return MyModel()

def GetInput():
    # Returns 5D tensor (B, C, D, H, W) compatible with 3D spatial padding
    return torch.rand(1, 2, 3, 4, 5, dtype=torch.float32)

