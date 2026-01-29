# torch.rand(N, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Emulate the ternary chain from the JIT bug example
        return torch.tensor(1, dtype=torch.int64) if x.numel() == 0 else (x[0] if x.numel() == 1 else x[1])

def my_model_function():
    return MyModel()

def GetInput():
    import random
    # Generate input with variable length 0-2 to test all branches
    length = random.choice([0, 1, 2])
    if length == 0:
        return torch.tensor([], dtype=torch.int64)
    else:
        return torch.randint(0, 10, (length,), dtype=torch.int64)

