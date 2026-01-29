# torch.rand(2, 4, 6, 6, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Weight shape matches the issue's description (out_channels=4, in_channels_per_group=1, kernel 3x3)
        self.weight = nn.Parameter(torch.rand(4, 1, 3, 3))
        # Non-contiguous bias (replicates original issue's scenario where bias is non-contiguous)
        self.bias_noncontig = nn.Parameter(torch.rand(8)[::2])  # Creates non-contiguous [4] tensor
        # Contiguous bias for comparison path
        self.bias_contig = nn.Parameter(torch.rand(4))  # Contiguous [4] tensor

    def forward(self, x):
        # Compute using non-contiguous bias (may trigger Winograd on ARM)
        out_winograd = F.conv2d(x, self.weight, self.bias_noncontig, groups=4)
        # Compute using contiguous bias (may trigger Slow2d on all platforms)
        out_slow = F.conv2d(x, self.weight, self.bias_contig, groups=4)
        # Return max absolute difference between outputs
        return (out_winograd - out_slow).abs().max()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 4, 6, 6, dtype=torch.float32)

