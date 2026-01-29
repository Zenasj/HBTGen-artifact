# torch.rand(8193, 50000, dtype=torch.float32), torch.rand(8193, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        H_aug, T = inputs
        return torch.pinverse(H_aug) @ T

def my_model_function():
    return MyModel()

def GetInput():
    m, n, k = 50000, 8193, 10  # Dimensions from the original reproduction code
    H_aug = torch.randn(n, m, dtype=torch.float32)  # Shape (8193, 50000)
    T = torch.randn(n, k, dtype=torch.float32)      # Shape (8193, 10)
    return (H_aug, T)

