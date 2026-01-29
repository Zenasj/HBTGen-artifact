# torch.rand(1, 1, 4, 4, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.full((16,), 3, dtype=torch.int32), requires_grad=False)  # Fixed parameter for bitwise operation
    
    def forward(self, x):
        # Flatten input to 1D tensor to match vectorized path conditions (size=16)
        x_flat = x.view(-1)
        return x_flat & self.b  # Bitwise AND operation that triggers vectorized code path

def my_model_function():
    return MyModel()

def GetInput():
    # Create 4D tensor with 16 elements (1x1x4x4) filled with value 6 (matches original issue's 'a' tensor)
    return torch.full((1, 1, 4, 4), 6, dtype=torch.int32)

