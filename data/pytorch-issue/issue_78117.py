# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape (1, 1, 1, 1)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, 1))  # Registered as parameter for quantization
        self.b = nn.Parameter(torch.randn(1))      # Registered as parameter for quantization

    def forward(self, x):
        # Reshape 4D input to 2D for linear layer compatibility
        x = x.view(x.size(0), -1)
        return torch.nn.functional.linear(x, self.w, self.b)

def my_model_function():
    # Returns model instance with properly initialized parameters
    return MyModel()

def GetInput():
    # Returns 4D tensor matching expected input shape
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

