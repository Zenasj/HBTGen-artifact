# torch.rand(5120000, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute problematic mean (float32, may suffer precision loss on CPU)
        mean_float = x.float().mean(dim=0)
        # Compute accurate reference using double precision
        mean_double = x.double().mean(dim=0)
        # Return absolute difference between the two methods
        return (mean_float - mean_double).abs()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the problematic scenario (large tensor with 3 channels)
    return torch.rand(5120000, 3, dtype=torch.float32)

