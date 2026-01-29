# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class X:
    def __init__(self):
        pass

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.jit.script_method
    def helper(self, x: X) -> X:
        return x  # This line triggers the JIT type resolution error
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dummy forward pass to ensure tensor input/output compatibility
        return x + 1.0

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

