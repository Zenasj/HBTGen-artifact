# torch.rand(1, 1, 512, 512, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3)  # Problematic kernel size 3

    def forward(self, x):
        # Compute outputs with and without gradients
        grad_output = self.conv(x)
        with torch.no_grad():
            no_grad_output = self.conv(x)
        # Return difference between outputs for comparison
        return grad_output - no_grad_output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 512, 512, dtype=torch.float32)

