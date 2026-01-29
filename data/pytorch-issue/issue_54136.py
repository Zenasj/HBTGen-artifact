# torch.randn(1, 1, 80, 140, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 16, 3, stride=1)  # Matches reported Conv2d configuration

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()  # Directly returns the model instance

def GetInput():
    return torch.randn(1, 1, 80, 140, dtype=torch.float32)  # Reproduces input shape from issue

