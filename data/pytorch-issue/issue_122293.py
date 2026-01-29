# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 2
        self.mode = "trilinear"
        self.align_corners = True

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the failing 5D input shape (batch, channels, depth, height, width)
    return torch.randn(2, 1, 8, 8, 8, dtype=torch.float32)

