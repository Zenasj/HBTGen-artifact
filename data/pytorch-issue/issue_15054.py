# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = 3  # Dilation factor from original issue's test case
        self.conv = nn.Conv2d(256, 256, 3, 1, padding=d, dilation=d, bias=False)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape
    return torch.rand(1, 256, 80, 45, dtype=torch.float32)

