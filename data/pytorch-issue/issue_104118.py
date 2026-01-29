# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # The problematic logspace call with complex end parameter
        a = torch.logspace(3, 10 + 3j, steps=5)
        return a

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input (not used in forward pass but required for model interface)
    return torch.rand(1)

