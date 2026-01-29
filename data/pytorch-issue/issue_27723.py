# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Spectral normalized layers to replicate the issue
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.fc = nn.utils.spectral_norm(nn.Linear(128 * 56 * 56, 10))  # Matches input shape after pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Down to 112x112
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Down to 56x56
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Returns a model instance with spectral normalization applied
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input requirements
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

