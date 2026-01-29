# torch.rand(1, 1, 1, 1, dtype=torch.float32)  # Dummy input tensor (unused in computation)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Use workaround for TorchScript's lack of "in" operator for strings
        # Original logic: return "a" in "abcd" → replaced with find() method
        return torch.tensor("abcd".find("a") != -1, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

