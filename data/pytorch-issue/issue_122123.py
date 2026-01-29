# torch.rand(32769, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 10))  # Matches first example's matrix multiplication dimensions
        
    def forward(self, x):
        # Slice first dimension (row-wise) from index 1 to 25 (exclusive)
        sliced = x[1:25, :]
        # Perform matrix multiplication with learned weights
        return sliced @ self.weight

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32769, 4, dtype=torch.float32)  # Matches input shape from first example

