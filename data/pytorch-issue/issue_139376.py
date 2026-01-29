# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Replicate the original function's operations
        y = torch.sum(torch.sum(x, dim=-1))  # Double reduction
        z = x / 10.0  # Pointwise division
        z_t = z.transpose(-2, -1).contiguous().transpose(-2, -1)  # Transpose/contiguous pattern
        return y, z, z_t

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random 4D tensor matching expected input shape (B,C,H,W)
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

