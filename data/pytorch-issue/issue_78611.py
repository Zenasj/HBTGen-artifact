# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape (1,2,3,4)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)  # Common layer for affected models

    def forward(self, x):
        # Reproduce the stride issue via reshape as described in the bug report
        x = x.reshape(x.shape)  # Triggers incorrect stride calculation
        # Apply convolution to demonstrate performance impact from memory format issues
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create channels_last input tensor with shape (1,2,3,4)
    return torch.randn(1, 2, 3, 4, dtype=torch.float32).to(memory_format=torch.channels_last)

