# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class Config:
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        # Compare with another Config instance to trigger __eq__ method
        other_config = Config()
        return x * 1.0 if self.config == other_config else x * 0.0

def my_model_function():
    config = Config()
    return MyModel(config)

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

