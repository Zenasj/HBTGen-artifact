# torch.rand(64, 1, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn
from torch import Tensor

class SimpleResidualBlock(nn.Module):
    def __init__(self, kernel_size: int, hidden_channels: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.batchnorm = nn.BatchNorm2d(num_features=hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, data: Tensor) -> Tensor:
        hidden = self.conv(data)
        hidden = self.batchnorm(hidden)
        hidden = self.relu(hidden)
        data = data + hidden
        return data

class MyModel(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        h: int,
        w: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pre_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resblocks = nn.Sequential(
            *(
                [
                    SimpleResidualBlock(kernel_size=kernel_size, hidden_channels=hidden_channels)
                    for _ in range(num_layers // 2)
                ]
                + [max_pool]
                + [
                    SimpleResidualBlock(kernel_size=kernel_size, hidden_channels=hidden_channels)
                    for _ in range(num_layers // 2)
                ]
                + [max_pool]
            )
        )
        self.linear = nn.Linear(
            in_features=(hidden_channels * (h // 4) * (w // 4)),
            out_features=num_classes,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, data: Tensor) -> Tensor:
        hidden = self.pre_conv(data)
        hidden = self.relu(hidden)
        data = data + hidden  # residual connection

        data = self.resblocks(data)

        data = self.flatten(data)
        data = self.linear(data)
        return data

def my_model_function():
    # Initialize with parameters from the original issue's example
    return MyModel(
        kernel_size=3,
        input_channels=1,
        hidden_channels=16,
        num_layers=8,
        h=28,
        w=28,
        num_classes=10,
    )

def GetInput():
    # Returns a random tensor with the expected input shape
    return torch.rand(64, 1, 28, 28, dtype=torch.float32)

