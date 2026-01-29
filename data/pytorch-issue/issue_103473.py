# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        out = torch.nn.functional.interpolate(
            x, size=(10, 20, 30), mode="trilinear", align_corners=True
        )
        return out + 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 3, 4, 5, dtype=torch.float32)

