# torch.rand(32, 512, 64, dtype=torch.float), torch.rand(32, 64, 128, dtype=torch.float)  # scaled by 1e4
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x1, x2 = inputs
        # Compute via batch matrix multiplication (torch.bmm equivalent)
        ans1 = x1 @ x2
        # Compute via explicit batched torch.mm
        ans2 = torch.stack([x1[i] @ x2[i] for i in range(x1.size(0))])
        # Calculate differences
        diff = ans2 - ans1
        return diff.abs().mean(), diff.abs().max()

def my_model_function():
    return MyModel()

def GetInput():
    B, M, K, N = 32, 512, 64, 128
    x1 = torch.randn(B, M, K) * 1e4  # Match original scaling
    x2 = torch.randn(B, K, N) * 1e4
    return (x1, x2)

