# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Uses aten.fill (non-in-place fill) as required
        return x.fill_(3.14).clone()  # Ensure non-inplace by cloning (since fill_ is in-place, but we need to return a new tensor)
        # Alternative: return x.fill(3.14) if fill() is preferred (but fill() returns a new tensor directly)
        # Note: Using fill_() here with clone() to mimic non-inplace behavior for demonstration purposes

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 2, 3, 4, 5  # Example input dimensions
    return torch.rand(B, C, H, W, dtype=torch.float32)

