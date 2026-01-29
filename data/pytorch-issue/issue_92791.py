# torch.rand(B, 3, 5, 5, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Based on the second test case's parameters with valid groups=1
        self.conv = nn.ConvTranspose3d(
            in_channels=3,  # Matches input channels
            out_channels=3,  # Matches weight's second dimension (3)
            kernel_size=3,  # From kernel dimensions in weight
            stride=1,  # From arg_4 in the first example
            padding=0,  # From arg_5
            output_padding=0,  # From arg_6
            groups=1,  # Fix non-positive groups issue
            bias=True,  # Matches presence of bias in first example (if included)
            dilation=1  # From arg_8
        )

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the second test case's input shape (5D)
    return torch.rand(1, 3, 5, 5, 5, dtype=torch.float32)

