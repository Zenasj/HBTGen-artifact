# torch.rand(B, 100, dtype=torch.float32)
import torch
import torch.nn as nn
import typing

class MyModel(nn.Module):
    def __init__(self, dims):
        super(MyModel, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x + torch.randn_like(x)

    def random_method(self):
        return torch.randn(self.dims)

    # Fix for PyCharm autocomplete issue as per the issue comments
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

def my_model_function():
    # Initialize with dims=100 as in the original example
    return MyModel(100)

def GetInput():
    # Matches input expected by forward() with batch size 1 and feature dim 100
    return torch.rand(1, 100, dtype=torch.float32)

