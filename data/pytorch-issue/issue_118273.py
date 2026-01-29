# torch.rand(B, 3, dtype=torch.float32)  # Inferred from example input array.array('i', [1, 2, 3])
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # Matches input size from example (3 elements)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Default batch size (can be adjusted)
    return torch.rand(B, 3, dtype=torch.float32)  # Matches inferred input shape

