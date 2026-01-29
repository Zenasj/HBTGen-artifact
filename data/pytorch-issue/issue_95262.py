# torch.rand(2, 3, dtype=torch.float16), torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        _ = y.copy_(x)  # In-place copy between different dtypes
        return torch.moveaxis(y, source=0, destination=1)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2, 3, dtype=torch.float16)
    y = torch.rand(2, 3, dtype=torch.float32)
    return (x, y)  # Tuple input matching MyModel's forward signature

