# torch.randint(0, 10, (N,), dtype=torch.int64)  # N is variable (e.g., 1-5 elements)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Replicates the original function's logic using a tensor input
        last_element = x[-1].item()
        length = len(x)
        return torch.rand(last_element, length)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 1D tensor with 1-5 elements (matching example input patterns)
    N = torch.randint(1, 6, (1,)).item()
    return torch.randint(0, 10, (N,), dtype=torch.int64)

