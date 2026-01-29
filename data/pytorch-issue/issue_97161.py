# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = x.data  # Segmentation Fault when vmap is applied
        print(y)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 3)

