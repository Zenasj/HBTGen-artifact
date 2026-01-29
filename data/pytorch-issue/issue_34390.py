# torch.rand(N, dtype=torch.float32)  # Input is a 1D tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = x.view(-1, 2)
        x = x.view(1, -1)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 1D tensor with even length to satisfy the view operations
    # Uses a random size between 0 and 10 (even numbers only) to test edge cases
    size = torch.randint(0, 6, (1,)).item() * 2  # Ensure even number of elements
    return torch.rand(size)

