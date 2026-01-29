# torch.rand(B, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x  # Model outputs raw values (not constrained to [0,1])

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor with values outside [0,1] to trigger the described issue
    return torch.rand(1, 1, dtype=torch.float32) * 3 - 1  # Random values in [-1, 2]

