# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        part1 = x[2:]
        part2 = x[:2]
        comp1 = part1 <= part2  # Buggy comparison without clone
        comp2 = part1.clone() <= part2  # Fixed comparison with clone
        return torch.any(torch.ne(comp1, comp2))  # Return True if outputs differ

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32)

