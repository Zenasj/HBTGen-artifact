# torch.randint(0, 10, (2, 2), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.randn(1).cuda())  # Initialize CUDA context

    def forward(self, x):
        return x  # Pass-through to ensure CUDA context is active

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (2, 2), dtype=torch.long)

