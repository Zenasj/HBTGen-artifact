# torch.rand(4, 1, 5, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Directly initialize parameters instead of using ParameterList to avoid DataParallel issues
        self.alpha0 = nn.Parameter(1e-3 * torch.randn(2, 5))
        self.alpha1 = nn.Parameter(1e-3 * torch.randn(3, 5))
        self.alpha2 = nn.Parameter(1e-3 * torch.randn(4, 5))
        self.alpha3 = nn.Parameter(1e-3 * torch.randn(5, 5))
        self.cnn = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        return x  # Minimal forward pass to test parameter replication

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 1, 5, 5, dtype=torch.float32)  # Batch, channels, H, W

