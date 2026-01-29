import torch
import torch.nn as nn

# torch.rand(B, 2, 3, 1, dtype=torch.float32)  # Inferred input shape based on test parameter dimensions
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the parameter shape used in the test case (2x3 tensor)
        self.weight = nn.Parameter(torch.randn(2, 3))
    
    def forward(self, x):
        # Example usage of the parameter to ensure it's part of the model's computation
        return x + self.weight.view(1, 2, 3, 1).expand_as(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching the inferred input shape (B=1, C=2, H=3, W=1)
    return torch.rand(1, 2, 3, 1, dtype=torch.float32)

