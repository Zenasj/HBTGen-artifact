# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x[..., :]

def my_model_function():
    return MyModel()

def GetInput():
    # Example input shape (2, 3, 4, 5) - arbitrary 4D tensor
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

