# torch.tensor(42, dtype=torch.uint32)  # Inferred input shape ()
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(42, dtype=torch.uint32)

