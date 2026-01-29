# torch.rand(B, C, H, dtype=torch.float32)  # 3D input with batch dimension constrained between 2-5
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate valid input within constraints (dim0 between 2-5)
    x = torch.rand(3, 3, 3, dtype=torch.float32)  # Valid batch size 3
    torch._dynamo.mark_dynamic(x, 0, min=2, max=5)  # Apply constraints to batch dimension
    return x

