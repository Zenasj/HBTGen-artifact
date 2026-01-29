# torch.rand(B, C, H, W, dtype=torch.float)  # e.g., (1, 3, 64, 64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)  # Added padding to match residual dimensions
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.iden = nn.Identity()  # Preserved from original code

    def forward(self, x):
        y = x
        y = self.iden(x)  # Redundant but kept to match original implementation
        x = self.conv(x)
        x = self.bn(x)
        x = torch.add(x, y)  # Residual connection
        x = self.relu(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Random input tensor with shape (B=1, C=3, H=64, W=64)
    return torch.rand(1, 3, 64, 64, dtype=torch.float)

