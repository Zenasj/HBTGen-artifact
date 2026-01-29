# torch.rand(B, 100, dtype=torch.float32)  # Input shape inferred from the original example (16, 100)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed mask as part of the model (1x100, first 10 elements True)
        mask = torch.zeros(1, 100, dtype=torch.bool)
        mask[0, :10] = True
        self.register_buffer('mask', mask)  # Ensure mask is not a parameter

    def forward(self, x):
        y = x.clone()  # Mimics original example where source is a clone of self
        # masked_scatter implementation (bug scenario)
        result1 = x.masked_scatter(self.mask, y)
        # torch.where-based "correct" implementation (desired behavior)
        result2 = torch.where(self.mask, y, x)
        # Return the absolute difference between the two methods
        return (result1 - result2).abs().sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected shape (B, 100)
    B = 2  # Arbitrary batch size (can be adjusted, 16 in original example)
    return torch.rand(B, 100, dtype=torch.float32)

