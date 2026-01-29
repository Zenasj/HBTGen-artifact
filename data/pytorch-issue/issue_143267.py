# torch.rand(2, 1, 1, 1, dtype=torch.complex64)
import torch
from torch import nn

class ComplexAbs32(nn.Module):
    def forward(self, x):
        return torch.abs(x.to(torch.complex32))

class ComplexAbs64(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model32 = ComplexAbs32()
        self.model64 = ComplexAbs64()

    def forward(self, x):
        out32 = self.model32(x)
        out64 = self.model64(x)
        difference = torch.abs(out32 - out64)
        return difference

def my_model_function():
    return MyModel()

def GetInput():
    data = torch.tensor(
        [[[[100 + 150j]]],
         [[[200 + 250j]]]],
        dtype=torch.complex64
    )
    return data

