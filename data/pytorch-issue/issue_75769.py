# torch.rand((), dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, A):
        ord_nograd = torch.rand([], dtype=torch.float64)  # Non-grad ord tensor
        ord_grad = torch.rand([], dtype=torch.float64).requires_grad_()  # Grad ord tensor
        
        # Compute first case (non-grad ord)
        try:
            result1 = torch.linalg.vector_norm(A, ord=ord_nograd)
        except Exception:
            result1 = None
        
        # Compute second case (grad ord, expected to error)
        try:
            result2 = torch.linalg.vector_norm(A, ord=ord_grad)
        except Exception:
            result2 = None
        
        # Discrepancy exists if one succeeds and the other fails
        discrepancy = (result1 is not None) and (result2 is None)
        return torch.tensor(discrepancy, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float64)

