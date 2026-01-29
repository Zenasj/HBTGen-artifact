# torch.rand(1, 105, 1, 1, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        in_channels = 105
        out_channels = 1
        kernel_size = [1, 1]
        stride = [14, 14]
        dilation = [1, 1]
        groups = 1
        padding = [8, 5, 9, 4]

        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, self.padding)
        return self.conv(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    ifm_shape = [1, 105, 1, 1]
    return torch.rand(ifm_shape, dtype=torch.float32, requires_grad=True)

