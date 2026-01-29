# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, xs_val):
        super().__init__()
        self.xs = xs_val  # Comparison value stored as integer
    
    def forward(self, x):
        if x.size(0) is self.xs:  # Trigger SymNodeVariable vs Python int comparison
            return x + 1
        else:
            return x * 2

def my_model_function():
    # Initialized with xs_val=2 to match test case scenario
    return MyModel(2)

def GetInput():
    # Returns 4D tensor with batch size 2 (to match xs_val=2)
    return torch.rand(2, 1, 1, 1, dtype=torch.float32)

