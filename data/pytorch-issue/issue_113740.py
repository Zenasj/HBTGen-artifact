# torch.rand(B, 80, 1, 30, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution parameters inferred from error logs:
        # in_channels=80, out_channels=480, kernel_size=(1,1)
        self.conv = nn.Conv2d(80, 480, kernel_size=(1, 1), stride=1, padding=0)
    
    def forward(self, x):
        # Reproduces the problematic convolution operation
        return self.conv(x)

def my_model_function():
    # Returns model instance with default initialization
    return MyModel()

def GetInput():
    # Generates input matching [B, 80, 1, 30] shape (B=10 from error logs)
    return torch.rand(10, 80, 1, 30, dtype=torch.float)

