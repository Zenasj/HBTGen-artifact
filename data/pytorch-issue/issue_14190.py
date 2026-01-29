# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare Float tensor with Long scalar (now works)
        other = torch.tensor(0, dtype=torch.long)
        comp_result = x > other  # Returns torch.bool tensor
        return comp_result

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a Float tensor matching the example's input shape
    return torch.rand(2, dtype=torch.float32)

