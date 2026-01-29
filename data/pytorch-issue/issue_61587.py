# torch.rand(100000, 32, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyBN1(nn.Module):
    def __init__(self, num_features):
        super(MyBN1, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Matches PyTorch's default epsilon

    def forward(self, x):
        # Compute batch statistics (mean and variance)
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)  # Matches PyTorch's training behavior
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn1d = nn.BatchNorm1d(32)  # Official PyTorch implementation
        self.mybn = MyBN1(32)           # Naive implementation from the issue

    def forward(self, x):
        # Return both outputs for external comparison
        return self.bn1d(x), self.mybn(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the issue's test case
    return torch.randn(100000, 32, device='cuda', dtype=torch.float32)

