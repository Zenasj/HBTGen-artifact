# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Corrected initialization without torch.jit.Attribute for eager mode
        self.foo = 0.1  # Previously: torch.jit.Attribute(0.1, float)
        assert 0.0 < self.foo  # Now works since self.foo is a float
        self.names_ages = {}  # Previously: torch.jit.Attribute({}, Dict[str, int])
        self.names_ages["someone"] = 20
        assert isinstance(self.names_ages["someone"], int)

    # Added forward method to satisfy model requirements
    def forward(self, x):
        return x * self.foo  # Example usage of self.foo in computation

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape comment: (B, C, H, W)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

