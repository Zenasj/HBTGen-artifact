# torch.rand(1, 3, dtype=torch.float32)  # Inferred input shape: 2D tensor with batch size 1 and 3 features
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Returns a fixed tensor of shape (1, 2), ignoring input x
        return torch.zeros(1, 2, dtype=x.dtype)  # Uses input dtype for consistency

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching expected input shape
    return torch.rand(1, 3, dtype=torch.float32)

