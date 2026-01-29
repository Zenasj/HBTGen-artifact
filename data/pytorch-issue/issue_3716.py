# torch.randn(1, 1, 500, 500, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Extract 2D matrix from 4D input (B, C, H, W)
        a = x[0, 0]  # Select first batch and channel
        
        # Compute QR decomposition using two methods
        # Method 1: Direct QR decomposition
        q1, r1 = torch.qr(a)
        
        # Method 2: GEQRF + ORGQR workflow
        m, tau = torch.geqrf(a)
        q2 = torch.orgqr(m, tau)
        r2 = torch.triu(m)  # R is the upper triangular part of M
        
        # Calculate maximum absolute differences
        q_diff = torch.max(torch.abs(q1 - q2)).item()
        r_diff = torch.max(torch.abs(r1 - r2)).item()
        
        # Return True if any difference exceeds 1e-3 threshold
        return torch.tensor(q_diff > 1e-3 or r_diff > 1e-3, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching expected 4D shape
    return torch.randn(1, 1, 500, 500, dtype=torch.float32)

