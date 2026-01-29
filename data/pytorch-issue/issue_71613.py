# torch.rand(20, 10, 2, dtype=torch.float32)  # Input shape for the complex tensor (batch_size, features, real_imag)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexLinear(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        linear = nn.Linear(dim, dim_out, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(linear.weight))
        self.bias = nn.Parameter(torch.view_as_real(linear.bias))

    def forward(self, x):
        weight = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias)
        return F.linear(x, weight, bias)

class ModReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1j * torch.angle(x))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ComplexLinear(10, 10),
            ModReLU()
        )

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random complex tensor input
    x = torch.view_as_complex(torch.rand(20, 10, 2)).to(torch.cfloat)
    return x

