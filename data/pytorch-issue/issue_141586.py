# torch.rand(B, C, H, W, dtype=torch.float32)  # Example input shape (e.g., B=1, C=3, H=32, W=32)
import torch
from torch import nn, Tensor

class MyModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * 2  # Example operation from the PR's type-checking demonstration

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)  # Matches 4D input expectation

