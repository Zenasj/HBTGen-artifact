# torch.rand(1, 3598, 4, 8, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D weight tensor causing MPS error when using F.linear
        self.projected = nn.Parameter(torch.rand(8))  # Shape [8]

    def forward(self, x):
        return F.linear(x, self.projected)

def my_model_function():
    # Returns model with a 1D weight parameter to reproduce the MPS error
    return MyModel()

def GetInput():
    # Generates input matching the model's expected dimensions
    return torch.rand(1, 3598, 4, 8)

