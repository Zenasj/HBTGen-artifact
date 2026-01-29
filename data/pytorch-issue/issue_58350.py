# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(10)  # Fixing typo from original issue's invalid 2nd argument
        self.conv = nn.LazyConv2d(10, 5, 5)  # Matches kernel_size=5, stride=5 from original issue
        
    def forward(self, x):
        # Process input through both lazy modules
        x = self.conv(x)  # Requires 4D input (B, C, H, W)
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        return self.linear(x)

def my_model_function():
    # Returns model with uninitialized parameters (as in original issue's setup)
    return MyModel()

def GetInput():
    # Returns 4D tensor compatible with Conv2d (B, C, H, W)
    return torch.rand(1, 3, 5, 5)  # Example shape: batch=1, channels=3, 5x5 spatial dims

