# torch.rand(1, 4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, output_size, kernel_size, stride, padding):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.fold(
            x,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

def my_model_function():
    return MyModel(
        output_size=(4, 4),
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
    )

def GetInput():
    return torch.randn(1, 4, 4)

