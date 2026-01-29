# torch.rand(B, dtype=torch.float32)  # Input is a 1D tensor of length B (must be even)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + 1

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random even length for the input tensor
    B = torch.randint(1, 10, (1,)).item() * 2
    return torch.rand(B, dtype=torch.float32)

