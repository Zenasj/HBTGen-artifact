# torch.rand(2, 256, 45, 18, dtype=torch.float32).to('cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, out_shape=(14, 24)):
        super().__init__()
        self.out_shape = out_shape  # Fixed output shape from the simplified test case

    def forward(self, x):
        # Perform nearest interpolation and multiply by zero to test gradient flow
        interpolated = F.interpolate(x, size=self.out_shape, mode='nearest')
        return interpolated * 0.0  # Creates zero gradient expectation

def my_model_function():
    # Returns the model with the problematic configuration
    return MyModel()

def GetInput():
    # Returns input tensor matching the simplified test case
    return torch.randn(2, 256, 45, 18, device='cuda', requires_grad=True)

