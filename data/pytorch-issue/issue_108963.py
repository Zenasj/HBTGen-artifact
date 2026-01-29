# torch.rand(1, dtype=torch.double)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed tensors used in computation
        self.register_buffer("tensor", torch.zeros(1, dtype=torch.double))
        self.register_buffer("index", torch.tensor([0], dtype=torch.long))

    def forward(self, source):
        # Reproduce the problematic computation path
        out = self.tensor.index_add(0, self.index, source, alpha=2.0) / 2.0
        return source, out  # Return tuple for comparison with uncompiled output

def my_model_function():
    # Returns the PyTorch model instance
    return MyModel()

def GetInput():
    # Generates a valid input tensor matching the model's requirements
    return torch.rand(1, dtype=torch.double)

