# torch.rand(B, 3, 5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 5, 1)  # Matches original conv layer
        self.flatten = nn.Flatten()         # Flattens 4D output to 2D
        # Bilinear layer with valid parameters (split 16 features into two parts)
        self.linear = nn.Bilinear(8, 8, 10)  # in1=8, in2=8, out=10

    def forward(self, x):
        x = self.conv(x)          # Output shape: (B, 16, 1, 1) for 5x5 input
        x = self.flatten(x)       # Flattened to (B, 16)
        x1, x2 = x.split([8, 8], dim=1)  # Split into two tensors for Bilinear
        return self.linear(x1, x2)  # Correct usage of Bilinear with two inputs

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 5, 5, dtype=torch.float32)  # Minimal input for 5x5 kernel

