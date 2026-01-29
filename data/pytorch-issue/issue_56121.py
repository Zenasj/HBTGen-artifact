# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: Batch=1, Channels=1, Height=64, Width=64
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Moved Conv2d initialization to __init__ to satisfy TorchScript requirements
        self.conv = nn.Conv2d(1, 3, 3, stride=1)  # 1 input channel, 3 output channels, 3x3 kernel

    def forward(self, x):
        return self.conv(x)  # Use pre-initialized module

def my_model_function():
    # Returns properly initialized model instance
    return MyModel()

def GetInput():
    # Generates valid input tensor matching expected shape (N, C, H, W)
    return torch.rand(1, 1, 64, 64, dtype=torch.float32)  # 64x64 image with 1 channel

