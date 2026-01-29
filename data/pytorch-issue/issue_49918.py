# torch.rand(B, C, H, W, dtype=torch.float32)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # Triggers mypy error due to None in tuple

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Example input with shape [2, 3, 5, 5] to match (B, C, H, W)
    return torch.rand(2, 3, 5, 5, dtype=torch.float32)

