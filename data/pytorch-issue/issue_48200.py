# torch.rand(B, 1, 10, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Squeeze channel dimension to match original 2D input shape (B, 10, 10)
        x = x.squeeze(1)  
        # Compute pairwise differences
        D = x[:, None, :] - x[:, :, None]  
        # Return mean of all elements
        return torch.mean(D)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching (B, C, H, W) with B=1, C=1, H=10, W=10
    return torch.rand(1, 1, 10, 10, dtype=torch.float32).cuda()

