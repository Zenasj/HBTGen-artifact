# torch.rand(B, C, H, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(6))  # channels=6, matches input shape's C dimension
        self.bias = nn.Parameter(torch.zeros(6))
        self.num_groups = 2  # Valid divisor of 6 (number of channels)

    def forward(self, x):
        return torch.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            eps=0,
            cudnn_enabled=False
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 6, 5, dtype=torch.float32)  # Matches (B, C, H) from issue's example

