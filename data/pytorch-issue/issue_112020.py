# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.init_value = torch.tensor([init_value], dtype=torch.float)
        self.register_buffer('radius', self.init_value, persistent=True)
    
    def forward(self, x):
        # Dummy forward to satisfy torch.compile requirements
        return x

def my_model_function():
    # Initialize with the original parameter value from the issue
    return MyModel(init_value=0.4)

def GetInput():
    # Return a minimal tensor matching the forward() input expectation
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

