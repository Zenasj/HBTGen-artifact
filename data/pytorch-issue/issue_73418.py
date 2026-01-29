# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Correct in_channels to 6 (matches input) and initialize weight properly
        self.conv2d = nn.Conv2d(6, 8, 3)
        # Initialize weights to match original intent (shape 8x6x3x3)
        self.conv2d.weight = nn.Parameter(torch.rand(8, 6, 3, 3))

    def forward(self, x):
        # Removed dynamic parameter modification (state_dict change during tracing)
        return self.conv2d(x)

def my_model_function():
    # Returns model with fixed initialization
    return MyModel()

def GetInput():
    # Matches the input shape from original code (1,6,64,64)
    return torch.rand(1, 6, 64, 64, dtype=torch.float32)

