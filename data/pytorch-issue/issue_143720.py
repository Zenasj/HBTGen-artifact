# torch.rand(1, 1, 1, dtype=torch.float)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding)

def my_model_function():
    # Initialize with parameters from the issue's example
    return MyModel(kernel_size=4, stride=3, padding=2)

def GetInput():
    # Input shape (B, C, L) = (1,1,1) as in the issue's example
    return torch.rand(1, 1, 1)

