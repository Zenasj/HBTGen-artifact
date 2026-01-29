# torch.randn(512, 768, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute delta with prepend
        prepend = torch.zeros(1, x.shape[1], dtype=x.dtype, device=x.device)
        delta = torch.diff(x, dim=0, prepend=prepend)
        cumsum = torch.cumsum(delta, dim=0)
        # Compute absolute difference and tolerance check
        diff = torch.abs(x - cumsum)
        tolerance = 1e-5 + 1e-5 * torch.abs(cumsum)
        mask = diff <= tolerance
        # Return whether all elements meet tolerance
        return mask.all()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(512, 768, dtype=torch.float32)

