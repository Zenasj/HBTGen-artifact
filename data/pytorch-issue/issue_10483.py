# torch.rand(1, 1, 1, 1000, dtype=torch.int64)  # Inferred input shape based on issue example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Flatten input to 1D tensor to match original example's operation pattern
        flat_x = x.view(-1)
        return flat_x.eq(flat_x).sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate 4D tensor matching expected input shape with compatible dtype
    return torch.randint(0, 5, (1, 1, 1, 1000), dtype=torch.int64)

