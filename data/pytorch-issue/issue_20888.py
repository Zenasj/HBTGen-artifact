# torch.rand(15000000, 45, dtype=torch.long, device='cuda')

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Generate permutation indices matching the first dimension size
        permutation = torch.randperm(x.size(0), device=x.device)
        # Perform reindexing operation that triggers the offset calculation bug
        return x[permutation]

def my_model_function():
    return MyModel()

def GetInput():
    # Create a large tensor on GPU with the problematic size and dtype
    return torch.empty(15000000, 45, device="cuda", dtype=torch.long).random_(0, 2**22)

