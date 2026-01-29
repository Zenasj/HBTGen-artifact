# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()  # Uses default in_features=10, out_features=5

def GetInput():
    # Returns a batch of 1 sample with 10 features (matches Linear layer's in_features)
    return torch.randn(1, 10)

