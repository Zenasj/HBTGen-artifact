# torch.rand(1, 32, 56, 56, dtype=torch.float)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = InvertedResidual(32, 32)

    def forward(self, x):
        return self.model(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_factor=1.):
        super(InvertedResidual, self).__init__()

        hidden_dim = int(in_channels * hidden_dim_factor)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 32, 56, 56, dtype=torch.float)

