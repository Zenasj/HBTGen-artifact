# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed parameter to replicate the non-grad tensor in the original example
        self.fixed = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def forward(self, x):
        # Replicate the addition operation from the issue's benchmark
        return self.fixed + x

def my_model_function():
    # Return model instance with default initialization
    return MyModel()

def GetInput():
    # Return input tensor matching the original benchmark's requires_grad=True tensor
    return torch.ones(1, requires_grad=True)

