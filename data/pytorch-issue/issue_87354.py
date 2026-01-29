# torch.rand(1, 32, 512, 512, 256, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Standard Conv3d with parameters from the issue
        self.conv = nn.Conv3d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        # LazyConv3d with parameters from the issue's comment
        self.lazy_conv = nn.LazyConv3d(
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
    
    def forward(self, x):
        # Run both models and check if outputs are all-zero (bug condition)
        out_conv = self.conv(x)
        out_lazy = self.lazy_conv(x)
        is_conv_zero = torch.all(out_conv == 0)
        is_lazy_zero = torch.all(out_lazy == 0)
        # Return True if both models output all zeros (indicating bug presence)
        return is_conv_zero and is_lazy_zero

def my_model_function():
    # Initialize the fused model
    return MyModel()

def GetInput():
    # Generate input tensor matching the issue's dimensions
    return torch.rand(1, 32, 512, 512, 256, dtype=torch.float32)

