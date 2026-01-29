# torch.rand(1, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Convert to float16 and back to int64 to detect precision loss
        converted = x.to(torch.float16)
        reconstructed = converted.to(torch.int64)
        # Return boolean tensor indicating mismatch
        return (reconstructed != x).any()

def my_model_function():
    return MyModel()

def GetInput():
    # Test case from the issue demonstrating precision loss
    return torch.tensor([2101], dtype=torch.int64)

