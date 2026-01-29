# torch.rand(1, 2, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # The model replicates the sum operation that exposed the MPS bug
        return x.sum(-3)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input with the problematic 5D shape (1,2,1,1,1)
    return torch.rand(1, 2, 1, 1, 1, dtype=torch.float32)

