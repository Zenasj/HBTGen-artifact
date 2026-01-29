# torch.rand(32768, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        sorted_values = x.sort().values
        unique_values = sorted_values.unique()
        return unique_values

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.full((32768,), -1, dtype=torch.int32)
    x[:100] = torch.iinfo(x.dtype).max
    return x

