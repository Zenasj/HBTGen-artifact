# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn
from torch.quasirandom import SobolEngine

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.se = SobolEngine(3)  # Initialization triggers SobolEngine crash if default type is CUDA

    def forward(self, x):
        return x  # Dummy forward to satisfy input/output requirements

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

