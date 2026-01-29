# torch.rand(1, dtype=torch.float32)  # Inferred input shape based on the function's requirement
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        # Compare the first element of the input tensor to 1 without using .item() to avoid JIT issues
        return a[0] == 1  # Returns a boolean tensor (0-dim) compatible with TorchScript

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 1D tensor of shape (1,) as the minimal input to access a[0]
    return torch.rand(1, dtype=torch.float32)

