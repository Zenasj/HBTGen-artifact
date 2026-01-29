# torch.rand(3, 3, dtype=torch.float64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute PyTorch's nuclear norm condition number
        pt_result = torch.linalg.cond(x, 'nuc')
        # Return whether the result is INF (indicating singular matrix, matching numpy's failure case)
        return torch.isinf(pt_result).all()

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random 3x3 tensor with float64 dtype (matches test case input dimensions)
    return torch.rand(3, 3, dtype=torch.float64)

