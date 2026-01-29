# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
    
    def forward(self, x):
        # Validate input dimensions using ndim
        if x.ndim != 4:
            raise ValueError("Input must be 4-dimensional (B, C, H, W)")
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching expected dimensions
    return torch.rand(2, 3, 32, 32)  # Batch=2, Channels=3, 32x32 spatial

