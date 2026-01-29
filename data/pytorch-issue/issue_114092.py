# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))

def my_model_function():
    return MyModel()

def GetInput():
    B = 64  # Matches CIFAR10 batch size from original code
    return torch.rand(B, 3, 32, 32, dtype=torch.float32)

