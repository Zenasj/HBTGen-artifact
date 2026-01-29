# torch.rand(B, C, H, W, dtype=...)  # Input shape: (10, 3, 8, 8)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.relu2(x)  # Second successive inplace ReLU
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(10, 3, 8, 8)

