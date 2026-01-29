# torch.rand(B, 2, 3, 2, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        contiguous_x = x.contiguous()
        op_contig = contiguous_x * 1.0  # Trigger computation as in issue
        op_noncontig = x * 1.0          # Same op on non-contiguous
        
        zeros_contig = torch.zeros_like(op_contig)
        zeros_noncontig = torch.zeros_like(op_noncontig)
        
        # Check if the non-contiguous zeros_like failed (returns non-zero)
        diff = torch.any(zeros_noncontig != 0)
        return diff.float().unsqueeze(0)  # Return 1.0 if bug exists

def my_model_function():
    return MyModel()

def GetInput():
    # Generate permuted tensor matching issue's input shape
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    x = torch.rand(1, 3, 2, 2, dtype=torch.float, device=device)
    return x.permute(0, 3, 1, 2)  # Permute to (B,2,3,2) as in issue

