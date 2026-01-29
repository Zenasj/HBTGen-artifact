# torch.rand(B, 1, dtype=torch.float32)  # Input shape: (n, 1), where n is the permutation length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        n = x.size(0)  # Extract n from the input's first dimension
        return torch.randperm(n)  # This line triggers the SymInt error when compiled with dynamic_shape=True

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a dummy tensor with shape (10, 1) to match the test case (n=10)
    return torch.rand(10, 1, dtype=torch.float32)

