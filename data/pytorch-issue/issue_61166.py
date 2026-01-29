# torch.rand(1, 1, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed Upsample implementation based on the PR's correction
        self.upsample = nn.Upsample(scale_factor=0.5, mode='bilinear')

    def forward(self, x):
        return self.upsample(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces the input shape and type from the issue's example
    return torch.rand(1, 1, 2, 2, dtype=torch.float32)

