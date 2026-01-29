# torch.rand(4, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        orig = torch.sgn(x)
        custom = torch.where(
            x.real != 0,
            torch.sign(x.real) + 0j,
            torch.sign(x.imag) + 0j
        )
        diff = torch.abs(orig - custom)
        return torch.all(diff < 1e-5)  # Return boolean tensor indicating similarity

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.complex64)

