# torch.rand(1, dtype=torch.float32)  # Input is a single-element tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float('nan')))  # Shared parameter initialized to NaN

    def forward(self, x):
        # Problematic approach: nan_to_num after multiplication (produces NaN gradient)
        problem_out = (x * self.w).nan_to_num(nan=0.0)
        # Fixed approach: nan_to_num before multiplication (produces 0 gradient)
        fixed_out = x * self.w.nan_to_num(nan=0.0)
        return problem_out, fixed_out  # Return both outputs for gradient comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, requires_grad=True)

