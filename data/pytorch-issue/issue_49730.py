# torch.rand(1, 3, 10, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.bn = nn.SyncBatchNorm(3, track_running_stats=False)  # Faulty configuration causing error

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 10, 10)  # Matches input shape (B, C, H, W)

