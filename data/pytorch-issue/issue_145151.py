# torch.randint(0, 2, (2000, 10), dtype=torch.int64)  # Inferred input shape and dtype from issue's example

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Execute both operations for memory comparison (as per issue's test)
        unique_out = torch.unique(x)  # Operation causing memory leak
        sort_out = torch.sort(x)[0]   # Control operation with stable memory
        return unique_out, sort_out  # Return both outputs for external analysis

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 2, (2000, 10), dtype=torch.int64)

