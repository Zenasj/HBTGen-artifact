# torch.rand(2, 3, 4, 5, dtype=torch.float32)  # Inferred input shape (B, C, H, W)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Uses torch.normal to trigger the 'normal_functional' operator
        return torch.normal(mean=x, std=torch.ones_like(x))

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 4D tensor matching the expected input shape
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

