# torch.rand(1)  # Dummy input tensor
import torch
from torch.utils.data import WeightedRandomSampler
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weights = torch.randn(10)  # Reproduces the issue's weights initialization
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=10)
    
    def forward(self, x):
        return x  # Dummy forward pass to satisfy model requirements

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Matches the dummy input expectation

