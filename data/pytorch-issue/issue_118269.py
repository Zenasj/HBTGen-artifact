# torch.rand(1000, 1000, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        real_var = torch.var(torch.real(x))
        imag_var = torch.var(torch.imag(x))
        total_var = torch.var(x)
        return real_var, imag_var, total_var

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1000, 1000, dtype=torch.complex64)

