# torch.randint(0, 2048, size=(), dtype=torch.int, device="cuda")  # Scalar integer input
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, n):
        condition = n >= 1024  # Boolean scalar tensor (shape ())
        x = torch.full((1,), condition, dtype=torch.bool, device="cuda")
        return x + 1  # Addition forces type conversion from bool to int

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random integer scalar on CUDA device
    return torch.randint(0, 2048, size=(), dtype=torch.int, device="cuda")

