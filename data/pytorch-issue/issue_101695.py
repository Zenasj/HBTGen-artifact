# torch.rand(1, 3, 64, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.convT = nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.min_val = min_value
        self.max_val = max_value

    def forward(self, x):
        v1 = self.convT(x)
        v2 = torch.clamp(v1, self.min_val, float('inf'))
        v3 = torch.clamp(v2, float('-inf'), self.max_val)
        return v3

def my_model_function():
    return MyModel(-0.5, 0.5)

def GetInput():
    return torch.randn(1, 3, 64, 64, dtype=torch.float32)

