# torch.rand(3, 3, dtype=torch.float32)  # Input shape inferred from the issue's 3x3 zero tensor example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute both norms as per the issue's comparison
        matrix_norm_out = torch.linalg.matrix_norm(x)
        norm_out = torch.linalg.norm(x)
        return matrix_norm_out, norm_out

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random 3x3 tensor with requires_grad=True to enable gradient comparison
    return torch.rand(3, 3, dtype=torch.float32, requires_grad=True)

