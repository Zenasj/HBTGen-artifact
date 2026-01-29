# torch.rand(B, 128, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 1)  # Matches in_features=128 and out_features=1

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a valid input tensor matching the Linear layer's in_features=128 requirement
    return torch.rand(302, 128, dtype=torch.float32)

