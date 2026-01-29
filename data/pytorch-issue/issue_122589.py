# torch.rand(2, 4, 3)
import torch
from torch import nn

class TestTensor(torch.Tensor):
    pass

class MyModel(nn.Module):
    def forward(self, x):
        return x.as_subclass(TestTensor)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 4, 3)

