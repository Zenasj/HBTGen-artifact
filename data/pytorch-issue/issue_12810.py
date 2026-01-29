# torch.rand(B, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x[None, :]  # Replicates the problematic slice with None

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5)  # Example 1D input tensor with shape (5,)

