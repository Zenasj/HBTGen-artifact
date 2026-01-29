# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Example input shape for image-like data
import torch
from torch import nn
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 3       # Input tensor channels (e.g., RGB)
    hidden_channels: int = 64     # Hidden layer channels
    kernel_size: int = 3          # Convolution kernel size

class MyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.input_channels, config.hidden_channels, config.kernel_size)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return x

def my_model_function():
    config = ModelConfig()
    return MyModel(config)

def GetInput():
    # Generate random input matching expected shape (B, C, H, W)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

