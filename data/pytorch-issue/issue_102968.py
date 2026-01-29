# torch.rand(10, 1, 128, 66, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 768, kernel_size=16, stride=10)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    # Initialize weights to match original test conditions
    # (weights are initialized randomly by default, which matches the issue's setup)
    return model

def GetInput():
    # Matches input shape from the issue's test case
    return torch.rand(10, 1, 128, 66, dtype=torch.float32)

