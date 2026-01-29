# torch.rand(B, C, H, dtype=torch.float32)
import torch
from torch import nn

class ModelWithoutMin(nn.Module):
    def forward(self, v1: torch.Tensor):
        return torch.randn_like(v1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alternative_path = ModelWithoutMin()  # submodule for comparison

    def forward(self, x: torch.Tensor):
        # Problematic path (with min operation)
        vx = x.min(dim=1).values  # shape becomes (B, H) where H=0
        output_problematic = torch.randn_like(vx)
        
        # Working path (without min, using vx as input)
        output_working = self.alternative_path(vx)
        
        # Return tuple to expose both outputs for error detection during compilation
        return (output_problematic, output_working)

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (10, 3, 0) as per the failing case in the issue
    return torch.rand(10, 3, 0, dtype=torch.float32)

