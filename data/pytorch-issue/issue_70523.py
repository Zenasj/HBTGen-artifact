# torch.randn(1, 64, 4480, 2976, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_non_grouped = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.conv_grouped = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
    
    def forward(self, x):
        out_non_grouped = self.conv_non_grouped(x)
        out_grouped = self.conv_grouped(x)
        return out_non_grouped, out_grouped  # Returns both outputs for comparison

def my_model_function():
    return MyModel(in_channels=64)  # Matches CH=64 from the issue example

def GetInput():
    return torch.randn(1, 64, 4480, 2976, dtype=torch.float32)

