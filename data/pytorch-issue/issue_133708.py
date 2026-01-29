import torch
import torch.nn as nn

# torch.rand(3, 5, dtype=torch.float32)  # Input shape (D1, D2)
class MyModel(nn.Module):
    def forward(self, x):
        # Compute offsets from input (sum over dimension 1 for 1D tensor)
        offsets = x.sum(dim=1)
        # Apply torch.diff (problematic operation with incorrect check)
        diffs = torch.diff(offsets)
        
        # Dynamic slicing based on input dimensions (D1, D2)
        D1, D2 = x.shape[-2], x.shape[-1]
        numerator = 5 * D1 - D2  # Could lead to negative values if D2 >5*D1
        denominator = D1 + D2
        idx = numerator // denominator
        
        # Slice using computed index (triggers slice_forward error)
        sliced = x[..., :idx]
        
        return sliced, diffs  # Return both outputs for validation

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 2D tensor with D1=3, D2=5 (safe slice index: 10//8=1)
    return torch.rand(3, 5, dtype=torch.float32)

