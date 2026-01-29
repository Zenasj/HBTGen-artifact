# torch.rand((), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize complex parameter with real/imag parts from uniform distribution
        self.a = nn.Parameter(torch.complex(torch.rand(3), torch.rand(3)))
    
    def forward(self, x):
        c = self.a * x  # Element-wise multiplication with scalar input
        # Problematic in-place clamping of real component
        c.real = c.real.clamp(0, 0.1)
        return c.abs().mean()  # Return mean of absolute values for backprop

def my_model_function():
    return MyModel()

def GetInput():
    # Scalar input tensor matching the original code's 'b' parameter
    return torch.tensor(1.0, dtype=torch.float32)

