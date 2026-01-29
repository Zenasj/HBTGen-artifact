# torch.rand(B, C, H, W, dtype=torch.float32)  # Example shape: (1, 2, 3, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)  # Problematic in_channels=1 case
        self.conv2 = nn.Conv2d(2, 1, 3)  # Working in_channels=2 case

    def forward(self, x):
        # Split input into 1-channel and 2-channel cases
        x1 = x[:, :1, :, :]  # Extract first channel for conv1
        y1 = self.conv1(x1)
        y2 = self.conv2(x)   # Use full 2 channels for conv2
        
        # Determine if the bug is present (y1 has no grad but y2 does)
        bug_present = not y1.requires_grad and y2.requires_grad
        return torch.tensor([bug_present], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 2-channel input with requires_grad=True to trigger the bug
    return torch.randn(1, 2, 3, 3, requires_grad=True, dtype=torch.float32)

