# torch.rand(1, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        # Use bilinear interpolation with align_corners=False as per original issue's model
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape (B, C, H, W)
    return torch.randn(1, 3, 256, 256, dtype=torch.float32)

