# torch.rand(B, 2, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)  # Split into two tensors along the second dimension
        a = a.squeeze(1)          # Reshape to (B, 1)
        b = b.squeeze(1)          # Reshape to (B, 1)
        # Compute distances using both methods
        d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
        d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        return d1 - d2  # Return the difference between the two methods

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Match batch size from examples
    return torch.rand(B, 2, 1, dtype=torch.float32) * 1e6  # Scale to large values to trigger numerical issues

