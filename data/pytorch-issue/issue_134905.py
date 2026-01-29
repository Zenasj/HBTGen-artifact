# Input: (A: torch.rand(3,3), B: torch.rand(3)), e.g., (torch.rand(3,3), torch.rand(3))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        A, B = inputs
        # Compute 32-bit solution
        A32 = A.float()
        B32 = B.float()
        try:
            sol32 = torch.linalg.solve(A32, B32)
        except:
            sol32 = torch.full_like(B32, float('nan'))
        
        # Compute 64-bit solution
        A64 = A.double()
        B64 = B.double()
        try:
            sol64 = torch.linalg.solve(A64, B64)
        except:
            sol64 = torch.full_like(B64, float('nan'))
        
        # Compute residuals
        residual32 = torch.sum((A32 @ sol32 - B32)**2)
        residual64 = torch.sum((A64 @ sol64 - B64)**2)
        
        # Return boolean indicating if 32-bit residual is orders of magnitude worse than 64-bit
        return torch.tensor([residual32 > 1e6 * residual64], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Ill-conditioned matrix example (condition number ~1e8)
    epsilon = 1e-8
    A = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0 + epsilon, 1.0],
        [1.0, 1.0, 1.0 + epsilon]
    ], dtype=torch.float64)
    B = torch.tensor([3.0, 3.0 + epsilon, 3.0 + epsilon], dtype=torch.float64)
    return (A, B)

