# torch.rand(N, 3), torch.rand(M, 3)  # N and M can be 0
import torch
import torch.nn as nn
import random

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        n = a.size(0)
        m = b.size(0)
        if n == 0 or m == 0:
            return torch.empty(n, m, dtype=a.dtype, device=a.device)
        else:
            return (a.unsqueeze(1) - b.unsqueeze(0)).norm(p=2, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tuple of two tensors with possible zero dimensions
    N = random.randint(0, 5)
    M = random.randint(0, 5)
    a = torch.rand(N, 3)
    b = torch.rand(M, 3)
    return (a, b)

