# torch.rand(10, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        result = torch.triu(x, diagonal=1)
        # Determine positions that should be zero (below diagonal=1)
        i, j = torch.meshgrid(
            torch.arange(x.size(0)),
            torch.arange(x.size(1)),
            indexing='ij'
        )
        mask_positions = (i > (j - 1))  # j < i+1 → positions below diagonal=1
        values = result[mask_positions]
        # Check for NaNs and zero values
        has_nans = torch.isnan(values).any()
        all_zero = torch.allclose(values, torch.zeros_like(values))
        correct = not has_nans and all_zero
        return torch.tensor([correct], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.full((10, 10), float('-inf'), dtype=torch.float32)

