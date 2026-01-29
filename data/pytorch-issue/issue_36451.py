# torch.rand(B, C, H, W, dtype=torch.half)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Cloning input to simulate two tensors for comparison
        a = x.clone()
        return torch.isclose(a, x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching expected input shape with half precision
    return torch.rand(1, 1, 1, 1, dtype=torch.half)

