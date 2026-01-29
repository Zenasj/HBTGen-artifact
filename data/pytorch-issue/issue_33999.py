# torch.rand(B, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.a = nn.Parameter(torch.randn(1))  # Matches the original Square model's parameter

    def forward(self, x):
        return (self.a * x) ** 2  # Forward computation as defined in the issue's Square class

def my_model_function():
    return MyModel()  # Returns an instance of the fused model (only one model exists here)

def GetInput():
    # Returns a random input matching the (B, 1) shape used in the issue's example (B=10)
    return torch.rand(10, 1, dtype=torch.float32)

