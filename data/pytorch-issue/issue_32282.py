# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (B, 16, 32, 128)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # n, c, h, w = x.shape
        # y = nn.functional.layer_norm(x, [c, h, w])       # not working
        # y = nn.functional.layer_norm(x, x.size()[1:])     # not working
        y = nn.functional.layer_norm(x, [16, 32, 128])
        return y

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 64
    C = 16
    H = 32
    W = 128
    return torch.randn(B, C, H, W)

