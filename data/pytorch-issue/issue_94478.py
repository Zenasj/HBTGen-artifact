# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, channels):
        super(MyModel, self).__init__()
        self.channels = channels
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(12)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 8, 12, 1, 1
    return torch.randn((B, C, H, W))

