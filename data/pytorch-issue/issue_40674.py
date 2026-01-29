# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Example input shape for a mini-batch of RGB images
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)  # Example layer
        self.fc = nn.Linear(16 * 222 * 222, 10)       # Example layer

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor with requires_grad=True to enable gradient computation
    return torch.randn(1, 3, 224, 224, requires_grad=True, dtype=torch.float32)

