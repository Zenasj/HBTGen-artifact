# torch.rand(2, dtype=torch.float64)
import torch
from torch import nn
import contextlib

@contextlib.contextmanager
def set_default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)

class MyModel(nn.Module):
    def forward(self, x):
        with set_default_dtype(torch.float64):
            return x * 2.0  # Example operation under context manager

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float64)  # Matches the model's expected input shape

