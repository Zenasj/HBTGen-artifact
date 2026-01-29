# torch.rand(1, dtype=torch.float32)  # Input is a 1-element tensor (for NaN comparison test)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare x with itself using two methods: torch.equal (no NaN equality) vs torch.allclose (with NaN handling)
        equal_result = torch.equal(x, x)
        allclose_result = torch.allclose(x, x, equal_nan=True)
        # Return results as a tensor of booleans (0/1)
        return torch.tensor([equal_result, allclose_result], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Create tensor with NaN while maintaining "randomness" as per issue's test case
    x = torch.rand(1)
    x[0] = float('nan')
    return x

