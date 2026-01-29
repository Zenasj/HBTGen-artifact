# torch.rand(3, dtype=torch.int64)  # Inferred input shape based on error log's FakeTensor size=(s0,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_cols=2):
        super().__init__()
        self.num_cols = num_cols  # Fixed parameter to avoid dynamic slice step issues

    def forward(self, offset):
        # Reproduces the problematic slicing logic using a fixed num_cols attribute
        return offset[:: self.num_cols]

def my_model_function():
    # Initialize with num_cols=2 as per original assertion in the repro
    return MyModel(num_cols=2)

def GetInput():
    # Generate a 1D integer tensor matching the offset's expected shape (inferred from error logs)
    return torch.randint(0, 10, (3,), dtype=torch.int64)

