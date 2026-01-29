# torch.rand(10, 10, dtype=torch.float32)  # Inferred input shape (batch_size=10, features=10)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        return (a_out, b_out)

def my_model_function():
    # Returns a CPU-based model instance (matches GetInput's default device)
    return MyModel()

def GetInput():
    # Returns a CPU tensor (matches model's default device unless moved)
    return torch.randn(10, 10, dtype=torch.float32)

