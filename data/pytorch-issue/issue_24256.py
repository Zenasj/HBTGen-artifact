# torch.rand(B, 2, dtype=torch.float32)  # Inferred input shape for the model
import torch
from torch import nn
from typing import Optional

class MyModel(nn.Module):
    def __init__(self, a: Optional[nn.Module] = None):
        super().__init__()
        self.has_a = a is not None
        self.a = a or nn.Identity()  # Use Identity if a is None
        if a is not None:
            self.b = nn.Linear(2, 2)  # Create b only when a is provided

    def forward(self, input):
        if not self.has_a:
            return input
        # Check existence of 'b' using hasattr to avoid attribute errors
        if hasattr(self, "b"):
            return self.a(self.b(input))
        else:
            return input  # Fallback path (should not be reached due to __init__ logic)

def my_model_function():
    # Return a model instance with default parameters (a=None)
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape
    return torch.rand(1, 2, dtype=torch.float32)  # Batch=1, features=2

