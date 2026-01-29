import torch
from torch import nn

# torch.rand(B, 3, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.original = OriginalNorm()
        self.fixed = FixedNorm()

    def forward(self, x):
        orig = self.original(x)
        fixed = self.fixed(x)
        return orig, fixed  # Return both outputs for comparison

class OriginalNorm(nn.Module):
    def forward(self, x):
        return x.norm(p=0.5)  # Original problematic norm computation

class FixedNorm(nn.Module):
    def forward(self, x):
        return (x.abs() + 1e-6).norm(p=0.5)  # Fixed version with epsilon

def my_model_function():
    return MyModel()

def GetInput():
    # Create input with a zero to trigger NaN gradients in original norm
    x = torch.rand(1, 3, dtype=torch.float32)
    x[0, 1] = 0.0  # Ensure a zero element
    x.requires_grad_(True)
    return x

