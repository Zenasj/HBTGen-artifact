# torch.randint(2**31 - 1, (1,), dtype=torch.int32)  # Input shape: (1,) near INT32_MAX
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Convert int32 to float32 and back to int32, check discrepancy
        float_converted = x.to(torch.float32)
        int_converted = float_converted.to(torch.int32)
        # Return boolean indicating mismatch (1 if discrepancy exists)
        return torch.any(x != int_converted).to(torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    # Create tensor with value near INT32_MAX (2^31-1)
    return torch.tensor([2**31 - 1], dtype=torch.int32)

