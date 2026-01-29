# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred from cudnn frontend JSON input tensor dimensions
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Conv layer parameters extracted from cudnn frontend JSON
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2  # Matches pre_padding/post_padding [2,2]
        )
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns the model with default initialization
    return MyModel()

def GetInput():
    # Generates input matching the cudnn frontend's input tensor dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

