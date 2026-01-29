# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hdim=256, depth=8):
        super().__init__()
        self.conv = nn.Conv2d(3, hdim, kernel_size=9, stride=4, padding=4)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hdim, hdim, kernel_size=3, padding=1, groups=hdim),
                nn.GELU(),
                nn.Conv2d(hdim, hdim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(hdim)
            ) for _ in range(depth)
        ])
    
    def forward(self, x):
        x = self.conv(x)
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return x

def my_model_function():
    # Returns a ConvMixer-like model instance with default parameters
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape (B, C, H, W)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

