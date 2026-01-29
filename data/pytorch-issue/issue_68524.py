# torch.rand(B, 1, dtype=torch.complex64)
import torch
import torch.nn as nn

class phi(nn.Module):
    def __init__(self):
        super(phi, self).__init__()
    def forward(self, x):
        real_part = torch.sin(x.real) * torch.cosh(x.imag)
        imag_part = torch.cos(x.real) * torch.sinh(
            torch.maximum(x.imag, -5 * torch.ones_like(x.imag))
        )
        return torch.complex(real_part, imag_part)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(1, 2, dtype=torch.complex64, bias=False)
        self.act = phi()
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 1, dtype=torch.complex64)

