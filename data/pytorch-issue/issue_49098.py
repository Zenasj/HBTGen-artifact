# torch.rand(B, C, H, W, dtype=torch.complex64)  # Complex input for angle() compatibility
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Apply angle to complex input, resulting in real tensor
        angle_out = torch.angle(x)
        # Max operation over dimension 1 (keepdim to maintain 4D shape)
        max_out, _ = torch.max(angle_out, dim=1, keepdim=True)
        # Clone to test forward AD for clone operation
        cloned = max_out.clone()
        return cloned

def my_model_function():
    return MyModel()

def GetInput():
    # Complex input tensor matching required 4D shape and dtype
    return torch.rand(2, 3, 4, 5, dtype=torch.complex64)

