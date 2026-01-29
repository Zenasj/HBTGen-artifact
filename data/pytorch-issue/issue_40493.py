# torch.rand(B, 5, 5, 10, dtype=torch.float32)  # Input shape (B, C=5, H=5, W=10)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Lazy modules infer input channels from first input
        self.conv1 = nn.LazyConv2d(4, 2)  # Output channels=4, kernel=2
        self.conv2 = nn.LazyConv2d(4, 2)
        self.linear = nn.LazyLinear(10)  # Output features=10

    def forward(self, x):
        x = self.conv1(x).clamp(min=0)  # ReLU equivalent
        x = self.conv2(x).clamp(min=0)
        # Flatten for linear layer
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input with shape (batch, 5, 5, 10)
    return torch.rand(1, 5, 5, 10, dtype=torch.float32)

