# torch.rand(2, 3, dtype=torch.float, device="cuda")  # Inferred input shape: matches the parameter's shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a parameter with the same shape as in the original issue's repro code
        self.param = nn.Parameter(
            torch.rand(2, 3, dtype=torch.float, device="cuda", requires_grad=True)
        )
    
    def forward(self, x):
        # Dummy forward pass to satisfy the input requirement (not used in the original issue's logic)
        return x

def my_model_function():
    # Returns an instance of MyModel with the parameter initialized
    return MyModel()

def GetInput():
    # Returns a random tensor compatible with MyModel's forward() input requirements
    return torch.rand(2, 3, dtype=torch.float, device="cuda")

