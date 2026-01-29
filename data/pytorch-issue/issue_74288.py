# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, (kernel_size, kernel_size),
                               padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, (kernel_size, kernel_size),
                               padding=kernel_size // 2, bias=False)
        self.norm1 = nn.BatchNorm2d(n_inner_channels)
        self.norm2 = nn.BatchNorm2d(n_channels)
        self.norm3 = nn.BatchNorm2d(n_channels)

    def forward(self, z, x=None):
        if x is None:
            x = torch.zeros_like(z)
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

def my_model_function():
    return MyModel(3, 1)  # Matches the ResBasicBlock(3,1) in the issue example

def GetInput():
    return torch.rand(2, 3, 32, 32)  # Batch size 2 to match the "two_input" test case

