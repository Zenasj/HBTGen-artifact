# torch.rand(B, 2, dtype=torch.float32)  # Inferred input shape based on example's 1D tensor of length 2
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        B, C = x.shape  # Assuming input is 2D (batch, features)
        index = torch.arange(C, device=x.device).unsqueeze(0).repeat(B, 1)  # Create index tensor [[0, 1], [0, 1], ...]
        x.scatter_(dim=1, index=index, value=0.0)  # Scatter to set all elements to 0.0
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a batched input tensor matching the model's expected shape
    B = 1  # Default batch size; can be any positive integer
    return torch.rand(B, 2, dtype=torch.float32)  # Matches the model's 2D input requirement

