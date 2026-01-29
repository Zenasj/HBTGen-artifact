# torch.rand(2, dtype=torch.float64)  # Inferred input shape: (2,) tensor with numerator and denominator
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        numerator, denominator = x.unbind()  # Split into numerator and denominator tensors
        # Compute division in float32 (PyTorch default) and cast to double for comparison
        a = (numerator.float() / denominator.float()).double()
        # Compute division in float64 (NumPy-equivalent precision)
        b = numerator.double() / denominator.double()
        # Return absolute difference between the two divisions
        return torch.abs(a - b)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns fixed example values from the issue to demonstrate discrepancy
    return torch.tensor([416.0, 1080.0], dtype=torch.float64)

