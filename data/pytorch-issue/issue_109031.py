# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, t):
        xs = ["bar", "foo", "baz"]
        # This line triggers the ListVariable index error in TorchDynamo
        return t + xs.index("foo")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

