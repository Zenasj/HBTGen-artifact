# torch.rand(B, C, H, W, dtype=torch.float32)  # B=1, C=4, H=5, W=5
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Valid groups=4 (input_channels=4 must divide groups, output_channels=8 must also divide groups)
        self.conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=4)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 4, 5, 5, dtype=torch.float32)

