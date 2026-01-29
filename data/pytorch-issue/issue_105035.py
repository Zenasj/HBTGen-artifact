# torch.rand((), dtype=torch.uint8)  # Shape: ()
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return (x + x).to(torch.int16)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(128, dtype=torch.uint8)

