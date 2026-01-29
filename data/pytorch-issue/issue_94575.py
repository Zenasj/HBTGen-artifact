# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (8, 640, 16, 16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, channels):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(640)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 8, 640, 16, 16
    return torch.randn((B, C, H, W))

