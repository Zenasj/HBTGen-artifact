# torch.rand(B, C, H, W, dtype=torch.complex64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand([3, 3, 3, 3], dtype=torch.complex64))
        self.bias = nn.Parameter(torch.rand([3], dtype=torch.complex64))
    
    def forward(self, x):
        # groups=0 is invalid (non-positive) but replaced with groups=1 to avoid runtime error
        return F.conv_transpose2d(
            x, self.weight, self.bias,
            stride=(2, 2), padding=2,
            output_padding=(1, 1), groups=1
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([1, 3, 4, 4], dtype=torch.complex64)

