# torch.rand(2, 2, 1, 1, dtype=torch.float32)  # Input shape: B=2, C=2, H=1, W=1

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 1, bias=False)
        self.layer2 = nn.Linear(1, 2, bias=False)
        # Initialize weights as per the issue's example
        self.layer1.weight = nn.Parameter(torch.tensor([[-0.0053, 0.3793]]))
        self.layer2.weight = nn.Parameter(torch.tensor([[-0.8230], [-0.7359]]))

    def forward(self, x):
        # Flatten input to 2D (B, features) for Linear layers
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected shape (B=2, C=2, H=1, W=1)
    return torch.rand(2, 2, 1, 1, dtype=torch.float32)

