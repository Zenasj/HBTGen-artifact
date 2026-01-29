# torch.rand(B, 2, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class ScaleTwo(nn.Module):
    def forward(self, x):
        return 2 * x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_two = ScaleTwo()  # Simple model (2*x)
        # ConvNet submodule with inferred parameters (input channels=2, kernel_size=1)
        self.conv = nn.Conv2d(2, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        scaled = self.scale_two(x)  # Run simple model
        conv_out = self.conv(x)     # Run convolutional model
        return scaled, conv_out     # Return outputs of both submodules

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size
    return torch.rand(B, 2, 1, 1, dtype=torch.float32)  # Matches input requirements for both submodels

