# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False,
            antialias=True
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Matches 4D input shape expected by interpolate (B, C, H, W)
    return torch.rand(1, 3, 800, 600)

