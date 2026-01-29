# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = (-1, 1, 0, 2)  # Dimensions as specified in the original issue

    def forward(self, x):
        return torch.sum(x, self.dim)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(128, 5, 24, 24)

