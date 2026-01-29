# torch.rand(5, 4, 5, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Exponent is a float64 tensor as in the issue example
        self.exponent = nn.Parameter(torch.rand(1, dtype=torch.float64))

    def forward(self, x):
        # Replicates the problematic torch.pow operation with dtype mismatch
        return torch.pow(x, self.exponent)

def my_model_function():
    # Returns model instance with exponent initialized as in the issue
    return MyModel()

def GetInput():
    # Returns input tensor with shape (5,4,5) and bfloat16 dtype
    return torch.rand(5, 4, 5, dtype=torch.bfloat16)

