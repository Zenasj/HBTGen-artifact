# torch.rand(1, 32, 512, 512, 256, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns a Conv3d model with 1x1 kernel (no padding/stride) and no bias
    return MyModel()

def GetInput():
    # Returns a 5D tensor matching (B, C, D, H, W) expected by Conv3d
    return torch.rand(1, 32, 512, 512, 256, dtype=torch.float32)

