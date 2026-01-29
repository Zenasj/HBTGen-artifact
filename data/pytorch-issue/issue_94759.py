# torch.rand(1, 2, 3, 3, dtype=torch.float32)
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3)
        self.conv = weight_norm(self.conv)  # Apply weight normalization
        self.conv.weight_g.data.zero_()     # Set weight_g to zero as per the issue's test case

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 3, 3, dtype=torch.float32)

