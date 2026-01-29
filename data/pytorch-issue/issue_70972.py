# torch.rand(B, 1, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Squeeze the channel dimension to process 2D matrices
        x_2d = x.squeeze(1)  # Convert to (B, H, W)
        upper = x_2d.triu(diagonal=0)
        lower = x_2d.tril(diagonal=0)
        return upper + lower  # Example output combining both operations

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 2, 2)  # Matches input shape (B, C, H, W) with C=1

