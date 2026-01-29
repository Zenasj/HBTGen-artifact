# torch.rand(2000, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2000, dtype=torch.float)

