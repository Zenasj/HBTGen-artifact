# torch.rand(B, 90, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(90, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2[:, :x1.size(1)] = x2[:, :x1.size(1)] + x1  # In-place addition of slices
        x3 = self.conv3(x2)
        return x3

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape used in the example (batch=3, channels=90, 5x6 spatial)
    return torch.randn(3, 90, 5, 6)

