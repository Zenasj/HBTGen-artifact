# torch.rand(32768, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Perform sort and check if sorted values match the expected (since input is arange(1,N))
        sorted_vals, _ = torch.sort(x)
        # Return 1.0 if correct (all elements match), 0.0 otherwise
        return torch.all(sorted_vals == x).float()

def my_model_function():
    return MyModel()

def GetInput():
    # Creates a tensor of size 32768 (exactly the threshold where the bug occurs)
    return torch.arange(1, 32769, dtype=torch.int64)

