# torch.rand(4, dtype=torch.float64, device="cuda")  # Inferred input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + 2e50  # Large constant causing Triton/float32 overflow in compiled path

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float64, device="cuda")  # Matches model's expected input

