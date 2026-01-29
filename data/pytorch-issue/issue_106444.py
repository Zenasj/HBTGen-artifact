# torch.rand(4, 6, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        first_column = x[:, 0]
        _, counts = first_column.unique(return_counts=True)
        return counts.max()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 6)  # Matches input shape from issue's failing example

