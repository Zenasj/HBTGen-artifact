# torch.rand(1, 12, 24, 24, dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.grouped_conv = nn.Conv2d(12, 12, 3, 1, 1, groups=12, bias=False)

    def forward(self, x):
        # Compute grouped convolution output
        out_grouped = self.grouped_conv(x)
        
        # Extract first group's kernel and input channel
        first_kernel = self.grouped_conv.weight.chunk(12, 0)[0]  # Split along output channels
        x_first = x[:, :1]  # First input channel
        
        # Compute split convolution for first group
        out_split_first = F.conv2d(x_first, first_kernel, padding=1)
        
        # Calculate maximum absolute difference between first group outputs
        return torch.max(torch.abs(out_grouped[:, :1] - out_split_first))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 12, 24, 24, dtype=torch.float32)

