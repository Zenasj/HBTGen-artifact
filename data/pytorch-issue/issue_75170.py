# torch.rand(B, C, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = torch.tensor([True, False, True, False], dtype=torch.bool)  # Example 1D mask for first dimension

    def forward(self, x):
        # Replicate the advanced indexing bug scenario
        return x[self.mask, True]  # Boolean mask combined with scalar boolean index

def my_model_function():
    return MyModel()

def GetInput():
    # Return 2D tensor matching the model's expected input shape
    return torch.rand(4, 3, dtype=torch.float32)  # Shape inferred from test case patterns

