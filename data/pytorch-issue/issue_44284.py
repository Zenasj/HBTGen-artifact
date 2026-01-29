# torch.rand(3, 3, dtype=torch.float32)  # Inferred input shape based on typical 2D tensor examples
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Example usage of torch.nonzero with explicit as_tuple=False to avoid warnings
        return torch.nonzero(x, as_tuple=False)

def my_model_function():
    # Returns a simple model using torch.nonzero; no weights needed
    return MyModel()

def GetInput():
    # Returns a random 3x3 tensor with float values (common nonzero test input)
    return torch.rand(3, 3, dtype=torch.float32)

