# torch.randn(B, 3, dtype=torch.float32), torch.complex64 tensor of shape (B, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.pow(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size, matches the issue's example
    x = torch.randn(B, 3, dtype=torch.float32)  # Real input (may include negatives)
    y_real = torch.randn(B, 3, dtype=torch.float32)
    y_imag = torch.randn(B, 3, dtype=torch.float32)
    y = torch.complex(y_real, y_imag)  # Complex input (dtype=torch.complex64)
    return (x, y)

