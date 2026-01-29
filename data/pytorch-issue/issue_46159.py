# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn
from torch._C import _disabled_torch_function_impl

class MyManifoldParameter(torch.nn.Parameter):
    __torch_function__ = _disabled_torch_function_impl

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = MyManifoldParameter(torch.randn(10, 5))
        self.bias = MyManifoldParameter(torch.randn(5))

    def forward(self, x):
        return x @ self.weight + self.bias

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 10)

