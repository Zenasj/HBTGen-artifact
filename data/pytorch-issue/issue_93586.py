# torch.rand(B, 1, H, W, dtype=torch.float64)  # Inferred input shape from CUDA float64 context and convolution tests
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)  # Standard 3x3 convolution with 1 input/2 output channels
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns a simple convolution model with double-precision weights
    model = MyModel()
    model.double()  # Explicitly set to float64 as per the test case
    return model

def GetInput():
    # Returns a 2x2 batch of 6x6 inputs (common small size for test cases)
    return torch.rand(2, 1, 6, 6, dtype=torch.float64)

