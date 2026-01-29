# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(10))  # Parameter being optimized

    def forward(self, x):
        # Dummy forward to accept input tensor (unused in loss computation)
        return x * self.param.sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a dummy input tensor matching the model's expected input shape
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

