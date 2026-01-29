# torch.rand(1024, 819, 3, dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Predefined mean values from the issue's subtraction operation
        self.register_buffer('mean', torch.tensor([107.0, 117.0, 123.0]))
    
    def forward(self, x):
        # Replicate the conversion and subtraction logic from the C++ code
        if x.dtype == torch.uint8:
            x = x.to(torch.float)
        return x - self.mean

def my_model_function():
    return MyModel()

def GetInput():
    # Create a random tensor matching the reported input shape [1024, 819, 3] with byte dtype
    return torch.randint(0, 256, (1024, 819, 3), dtype=torch.uint8)

