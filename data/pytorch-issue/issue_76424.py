# torch.rand(B, in_features, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features=10, out_features=20):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)  # Default slope as inferred

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

def my_model_function():
    # Returns an instance with default parameters (in_features=10, out_features=20)
    return MyModel()

def GetInput():
    # Returns a random input tensor with shape (1, 10) matching the default model
    return torch.rand(1, 10, dtype=torch.float)

