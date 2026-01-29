# torch.rand(3, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        # Extract upper triangular indices based on input shape
        row, col = x.shape[-2:]
        indices = torch.triu_indices(row, col, offset=self.offset, dtype=torch.long)
        return x[indices[0], indices[1]]

def my_model_function():
    # Default offset from the repro example (offset=0)
    return MyModel(offset=0)

def GetInput():
    # Generate random 2D input matching the repro's row/col range (3-6)
    row = torch.randint(3, 7, (1,)).item()
    col = torch.randint(3, 7, (1,)).item()
    return torch.rand(row, col, dtype=torch.float32)

