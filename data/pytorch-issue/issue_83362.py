# torch.rand(4, 576, 768, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(768))  # Initialized for valid computation
        self.bias = nn.Parameter(torch.randn(768))
        self.normalized_shape = [768]
        self.eps = 0.01

    def forward(self, x):
        # native_layer_norm returns a tuple (y, mean, variance), return only the output tensor
        return torch.ops.aten.native_layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )[0]

def my_model_function():
    return MyModel()

def GetInput():
    # Create input matching the issue's problematic shape/stride (contiguous tensor)
    return torch.rand((4, 576, 768), dtype=torch.float32)

