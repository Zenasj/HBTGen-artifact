# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, dilation=1, groups=1)
        
    def forward(self, x):
        return self.conv1(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 16, 128, 28, 28
    return torch.randn(B, C, H, W, dtype=torch.float, device='cuda', requires_grad=True)

