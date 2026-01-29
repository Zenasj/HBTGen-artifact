# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the error by calling torch.normal with explicit layout=torch.strided
        return torch.normal(2.0, 2.0, size=(2, 2, 2), dtype=torch.bfloat16, layout=torch.strided)

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input tensor (not used by the model but required for interface)
    return torch.rand(1)

