# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(MyConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )

    def forward(self, inputs, mask=None):
        outputs = super(MyConv1d, self).forward(inputs)
        return outputs

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = MyConv1d(4, 4, kernel_size=4)  # From original issue's MyConv1d
        self.conv_transpose2d = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=1)  # From comment's example

    def forward(self, x):
        # Process Conv1d: reshape 4D input to 3D (B, C, L)
        x_1d = x.view(x.size(0), x.size(1), -1)  # e.g., (B,4, 4*4) = (B,4,16)
        out1 = self.conv1d(x_1d)
        # Process ConvTranspose2d: use original 4D input
        out2 = self.conv_transpose2d(x)
        return out1, out2  # Return both outputs for Dynamo testing

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 4, 4, 4, dtype=torch.float32)  # 4D input compatible with both modules

