# torch.rand(2, 2, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2, device='cuda'), requires_grad=False)  # Match original weight setup

    def forward(self, x):
        H, tau = torch.geqrf(self.weight)
        H.requires_grad_(True)
        tau.requires_grad_(True)
        
        # Original ORMQR implementation (backward not implemented)
        y_ormqr = torch.ormqr(H, tau, x, left=False)
        
        # Alternative differentiable implementation using householder_product
        Q = torch.linalg.householder_product(H, tau)
        y_alt = Q.transpose(-2, -1) @ x  # Matches left=False behavior
        
        return y_ormqr, y_alt  # Return both outputs for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2, requires_grad=True, device='cuda')

