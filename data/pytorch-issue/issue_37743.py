import math
import torch
import torch.nn as nn

# torch.rand(1, dtype=torch.float32)  # Input is a single-element float32 tensor
class MyModel(nn.Module):
    def forward(self, x):
        rem = torch.remainder(x, math.pi)
        fmod = torch.fmod(x, math.pi)
        # Return True if remainder and fmod results differ beyond 1e-4 tolerance
        return torch.tensor([not torch.allclose(rem, fmod, atol=1e-4)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([1e9], dtype=torch.float32)

