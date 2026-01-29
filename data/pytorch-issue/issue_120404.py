# torch.randn(10, 6, dtype=torch.float32) * 123456000
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = 3  # Fixed parameter from the original code's "left = 3"
    
    def forward(self, x):
        # Compute mean of full columns slice then take first 'left' elements
        batched_mean = x.mean(0)[:self.left]  # Equivalent to x[:, :columns].mean(0)[:left]
        # Compute mean of first 'left' columns directly
        sliced_mean = x[:, :self.left].mean(0)
        # Return difference between the two means
        return batched_mean - sliced_mean

def my_model_function():
    return MyModel()

def GetInput():
    # Generate scaled random input matching the original experiment's parameters
    return torch.randn(10, 6, dtype=torch.float32) * 123456000

