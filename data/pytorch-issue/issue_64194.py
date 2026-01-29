# torch.rand(B, 512, 1024, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the scenario where reduction on small dimensions caused CUDA config errors
        return torch.mean(x)  # Example operation that triggers reduction kernels

def my_model_function():
    # Returns model instance with default PyTorch initialization
    return MyModel()

def GetInput():
    # Generates input matching the problematic scenario from the issue
    return torch.rand(1, 512, 1024, 3, dtype=torch.float32)

