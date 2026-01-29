# torch.rand(2, 3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Tensor-based min value for second clamp operation
        self.min_tensor = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x):
        # First clamp with scalar bounds
        a = torch.clamp(x, min=0.2, max=1.0)
        # Second clamp with tensor min value
        b = torch.clamp(a, min=self.min_tensor)
        # Third clamp with scalar min (no max)
        c = torch.clamp(b, min=0.1)
        return c

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape and dtype from the original test script
    return torch.randn(2, 3, 4, dtype=torch.float32)

