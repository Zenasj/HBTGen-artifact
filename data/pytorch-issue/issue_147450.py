# torch.rand(1, 1, 1, 1, dtype=torch.float32)  # Input shape inferred from issue example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input):
        # Compute polygamma(n=1, input) and return whether the output is finite (captures discrepancy)
        return torch.isfinite(torch.special.polygamma(1, input))

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input in range [-2.0, 0.0] to trigger edge cases near the problematic value (-1.0)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32) * 2 - 2

