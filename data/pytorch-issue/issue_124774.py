# torch.rand(1, 1, 3, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = 4312491  # Integer causing aliasing issue in forward AD

    def forward(self, x):
        return self.multiplier * x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape expected by MyModel (1x1x3x1 tensor)
    return torch.rand(1, 1, 3, 1, dtype=torch.float32)

