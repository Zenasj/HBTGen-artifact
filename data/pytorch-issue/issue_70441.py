# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModuleList(nn.ModuleList):
    def __add__(self, other):
        if not isinstance(other, MyModuleList):
            raise TypeError(f"ModuleList concatenation should only be used with another MyModuleList instance, but got {type(other).__name__}")
        concat = MyModuleList()
        offset = len(self)
        for i in range(len(self)):
            concat.add_module(str(i), self[i])
        for j in range(len(other)):
            concat.add_module(str(offset + j), other[j])
        return concat

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create two ModuleLists and concatenate them
        a = MyModuleList([nn.Linear(10, 20)])
        b = MyModuleList([nn.ReLU()])
        self.layers = a + b  # Uses custom __add__ operator

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 10)  # Batch size 5, input features 10

