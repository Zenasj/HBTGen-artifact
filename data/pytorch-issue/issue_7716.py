# torch.rand(B, C, dtype=torch.float32)  # Input shape (Batch, Channels), B >= 2 to avoid batchnorm error
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(3)  # Matches the example's channel count (3)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor of shape (2, 3) to avoid batch size 1 error
    return torch.rand(2, 3, dtype=torch.float32)

