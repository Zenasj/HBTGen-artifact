# torch.rand(B, 1, 1, 5, dtype=torch.complex64)  # Inferred input shape based on the issue's example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Cast complex tensor to float without warning (as per the original bug)
        return x.float()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 4D tensor matching the input shape comment (B, C, H, W)
    return torch.rand(1, 1, 1, 5, dtype=torch.complex64)

