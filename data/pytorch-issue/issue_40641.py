# torch.rand(B, C, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        size = x.size()
        a: torch.Size = size[:1]  # This line triggers the type error discussed
        return x  # Dummy return to satisfy nn.Module requirements

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float32)  # Example input (B=2, C=3)

