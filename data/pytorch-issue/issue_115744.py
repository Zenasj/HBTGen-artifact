# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 3, bias=False)
        parametrize.register_parametrization(self.layer, "weight", Symmetric())
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)

