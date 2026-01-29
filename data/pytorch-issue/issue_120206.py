# torch.rand(1, 1, 2, 2, dtype=torch.float32)  # Inferred input shape (batch, channel, height, width)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal model to process input tensor; focus on __repr__ behavior in FakeTensorMode
    def forward(self, x):
        return x  # Pass-through to trigger __repr__ during FakeTensorMode conversion

def my_model_function():
    return MyModel()

def GetInput():
    indices = torch.tensor([
        [0, 0],  # Batch indices
        [0, 0],  # Channel indices
        [0, 1],  # Row indices
        [1, 0]   # Column indices
    ], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(1, 1, 2, 2))

