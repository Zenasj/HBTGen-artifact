# torch.rand(2, 100, dtype=torch.float32)  # Inferred input shape from the issue's real_inputs
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.poisson(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 100, dtype=torch.float32)

