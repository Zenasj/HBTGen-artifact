# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Bilinear upsample triggers symbolic expressions causing the codegen bug
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 3, 32, 32  # Minimal input shape to trigger symbolic expressions
    return torch.rand(B, C, H, W, dtype=torch.float32)

