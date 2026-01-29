# torch.rand(5, dtype=torch.float32)
import torch
from torch import nn

class DoNothing:
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MyModel(nn.Module):
    def forward(self, x):
        with DoNothing():
            x = x * 2
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, device="cuda")

