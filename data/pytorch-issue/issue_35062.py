# torch.rand(0, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim  # Dimension to apply reduction (matches test case)

    def forward(self, x):
        # Compute max and min along specified dimension for comparison
        max_val, _ = x.max(self.dim, keepdim=False)
        min_val, _ = x.min(self.dim, keepdim=False)
        return max_val, min_val

def my_model_function():
    # Returns an instance with default dimension (1) as per test cases
    return MyModel()

def GetInput():
    # Returns zero-sized tensor matching test input shape (0,4)
    return torch.rand(0, 4, dtype=torch.float32)

