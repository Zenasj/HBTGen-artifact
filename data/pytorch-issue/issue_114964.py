# torch.rand(B, 1, dtype=torch.float)
import dataclasses
import torch
from torch import nn

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    a: int = 1  # Matches frozen dataclass structure from issue example

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Triggers __setattr__ in frozen dataclass __init__
        self.linear = nn.Linear(config.a, 10)  # Uses config attribute

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    config = ModelConfig()  # Creates frozen dataclass instance
    return MyModel(config)

def GetInput():
    return torch.rand(1, 1, dtype=torch.float)  # Matches input shape (B, a)

