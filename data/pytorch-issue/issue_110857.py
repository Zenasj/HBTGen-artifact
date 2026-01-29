# torch.rand(1, 3, 20, 20, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns a Conv2d model with 3 input channels, 3 output channels, 3x3 kernel
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 20, 20, dtype=torch.float32)

