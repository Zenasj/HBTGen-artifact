# torch.rand(B, C, dtype=torch.float16)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, in_features=6, out_features=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))

    def forward(self, x):
        # Compute standard linear output
        standard = F.linear(x, self.weight)
        
        # Split input and weights for distributed computation
        x1, x2 = x.split([3, 3], dim=1)  # Split input features into two parts
        w1, w2 = self.weight.split([3, 3], dim=1)  # Split weights' columns into two parts
        
        # Compute distributed outputs and sum them
        part1 = F.linear(x1, w1)
        part2 = F.linear(x2, w2)
        distributed = part1 + part2
        
        # Return the difference between standard and distributed outputs
        return standard - distributed

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 6, dtype=torch.float16)

