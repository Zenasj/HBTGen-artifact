# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Inferred input shape from common image-like data and MPS compatibility
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Core layers using weight_norm to trigger the reported bug scenario
        self.conv1 = weight_norm(nn.Conv2d(3, 16, kernel_size=3, padding=1))  # Matches input channel count
        self.fc = nn.Linear(16 * 32 * 32, 10)  # Flatten spatial dimensions (assuming no pooling)

    def forward(self, x):
        x = self.conv1(x)  # Example forward pass using weight-norm layer
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        return self.fc(x)

def my_model_function():
    # Returns model instance with weight normalization (critical component in the issue)
    return MyModel()

def GetInput():
    # Generates input matching (B, C, H, W) with MPS-compatible float32 dtype
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

