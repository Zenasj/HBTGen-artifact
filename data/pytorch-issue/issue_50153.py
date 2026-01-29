# torch.rand(8, 64, 64, 64, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=1,
            groups=1,
            bias=False
        ).to(torch.float16)  # Matches half() conversion in original code

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(8, 64, 64, 64, dtype=torch.float16, requires_grad=True)

