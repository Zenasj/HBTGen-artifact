# torch.rand(32, 100, 3, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        sum_val = torch.sum(x, dim=(), keepdim=False)
        mean_val = torch.mean(x, dim=(), keepdim=False)
        return sum_val, mean_val  # Return both for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 100, 3, dtype=torch.float32, requires_grad=True)

