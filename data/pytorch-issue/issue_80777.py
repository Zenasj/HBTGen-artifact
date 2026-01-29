# torch.rand(B, 4, 96, 96, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Create tensor with added leading dimension and perform assignment
        out = torch.zeros((x.size(0), 1, *x.shape[1:]), dtype=x.dtype, device=x.device)
        out[:, 0] = x  # Replicate the core assignment operation from the example
        return out

def my_model_function():
    # Returns model instance that reproduces the small-operation threading issue
    return MyModel()

def GetInput():
    # Returns tensor matching the expected input shape (B=1, C=4, H=96, W=96)
    return torch.rand(1, 4, 96, 96, dtype=torch.float32)

