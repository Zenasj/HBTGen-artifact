# torch.rand(10, 3, 3, dtype=torch.complex64)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Symmetrize the input tensor
        x = 0.5 * (x.transpose(-1, -2) + x)
        # Apply matrix exponential to the imaginary part
        x.imag = torch.matrix_exp(x.imag)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random complex tensor with the required shape
    torch.manual_seed(42)
    tens = 0.5 * torch.randn(10, 3, 3, dtype=torch.complex64)
    return tens

