# torch.rand(B, C, H, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Apply nearest-neighbor interpolation to expand last dimension from 5 → 11
        v = F.interpolate(x, [11], mode="nearest")
        # Compute 2nd smallest value (k=2) across flattened tensor
        return torch.kthvalue(v, 2)

def my_model_function():
    return MyModel()

def GetInput():
    # Create 3D tensor matching the original issue's input shape (5,5,5)
    return torch.rand(5, 5, 5, dtype=torch.float32)

