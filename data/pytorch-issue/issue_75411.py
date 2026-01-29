# torch.rand(12, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        out_float = torch.tanh(x)
        out_double = torch.tanh(x.double()).float()
        # Return 1.0 if outputs match within tolerance, 0.0 otherwise
        return torch.tensor(torch.allclose(out_float, out_double, atol=1e-4), dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the original issue's value range (10-50)
    return torch.rand(12, 1, dtype=torch.float32) * 40 + 10

