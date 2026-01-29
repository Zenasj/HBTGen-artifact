# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Apply torch.unique along the second dimension (dim=1)
        # Represents usage of unique operation affected by binding changes
        return torch.unique(x, dim=1)

def my_model_function():
    # Returns a model instance using the unique operation
    return MyModel()

def GetInput():
    # Generate a 4D tensor (B=2, C=3, H=4, W=5) as a valid input
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

