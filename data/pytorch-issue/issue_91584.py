# torch.rand(B, 1, 4, 6, 7, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        splits = torch.tensor_split(x, 2, 2)  # Split along dim=2 into 2 parts
        return splits[0] if splits else torch.empty(0)  # Return first split or empty tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a large input tensor (B=1000) to simulate memory-intensive scenarios
    return torch.rand(1000, 1, 4, 6, 7, dtype=torch.float64)

