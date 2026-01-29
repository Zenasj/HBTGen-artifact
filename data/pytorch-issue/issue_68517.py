# Input: (torch.rand(B, 1, dtype=torch.float32), torch.rand(B, 1, dtype=torch.float32))
import torch
from torch import nn

class E1Module(nn.Module):
    def forward(self, a, b):
        return a**2 - 2 * a * b + b**2

class E2Module(nn.Module):
    def forward(self, a, b):
        return torch.square(a - b)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = E1Module()  # Submodule for E1 = a² - 2ab + b²
        self.e2 = E2Module()  # Submodule for E2 = torch.square(a - b)

    def forward(self, inputs):
        a, b = inputs
        e1_out = self.e1(a, b)
        e2_out = self.e2(a, b)
        # Return boolean tensor indicating if outputs are close within default tolerances
        return torch.isclose(e1_out, e2_out)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Default batch size (adjustable)
    a = torch.rand(B, 1, dtype=torch.float32)
    b = torch.rand(B, 1, dtype=torch.float32)
    return (a, b)  # Returns tuple of two tensors matching the input shape

