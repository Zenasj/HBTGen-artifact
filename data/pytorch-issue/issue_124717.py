# torch.rand(4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.disabled = False  # Attribute controlling adapter behavior

    def forward(self, x):
        out = x * x
        if self.disabled:
            return out
        out = out + 1
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 4)

