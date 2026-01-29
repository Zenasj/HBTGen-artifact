# torch.rand(2, 1000, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, units=1000):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(units, units))

    def forward(self, x):
        # Compute outputs for batch size 1 (first sample) and batch size 2 (full input)
        out_single = F.linear(x[:1], self.weights)
        out_double = F.linear(x, self.weights)
        # Calculate mean absolute difference between first elements of outputs
        diff = (out_single[0] - out_double[0]).abs().mean()
        return diff  # Return the error as a tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Create input with two identical samples (batch size 2)
    return torch.rand(1, 1000).repeat(2, 1)

