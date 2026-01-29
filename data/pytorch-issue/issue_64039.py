# torch.rand(B, 2, 3, dtype=torch.double)
import torch
from torch import nn

class BatchNorm1dModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.bnorm = nn.BatchNorm1d(feat_dim)
        self.linear1 = nn.Linear(feat_dim, feat_dim)
        self.linear2 = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bnorm(x.transpose(1,2)).transpose(1,2)
        x = self.linear2(x)
        return x

class BatchNorm2dModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.bnorm = nn.BatchNorm2d(feat_dim)
        self.linear1 = nn.Linear(feat_dim, feat_dim)
        self.linear2 = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = x.unsqueeze(-1)
        x = self.bnorm(x.transpose(1,2)).transpose(1,2)
        x = x.squeeze(-1)
        x = self.linear2(x)
        return x

class MyModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.model1d = BatchNorm1dModel(feat_dim)
        self.model2d = BatchNorm2dModel(feat_dim)
    
    def forward(self, x):
        out1 = self.model1d(x)
        out2 = self.model2d(x)
        return (out1, out2)  # Returns both outputs for comparison

def my_model_function():
    # Initialize with feat_dim=3 and double precision as in original code
    return MyModel(feat_dim=3).double()

def GetInput():
    # Returns a tensor matching the input expected by MyModel
    return torch.randn(2, 2, 3, requires_grad=True, dtype=torch.double)

