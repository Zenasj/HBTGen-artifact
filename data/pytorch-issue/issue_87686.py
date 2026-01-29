# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, 1))  # Matches the example's weight shape

    def forward(self, x):
        # Use positional arguments to avoid keyword-argument quantization issues
        return F.linear(x, self.w)

def my_model_function():
    return MyModel()

def GetInput():
    # 4D tensor to match B,C,H,W convention (collapsed to 2D via 1x1 spatial dims)
    return torch.rand(1, 1, 1, 1)

