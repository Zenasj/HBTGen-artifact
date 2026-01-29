# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # SAM's pixel mean and std (example values from common practice)
        self.pixel_mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]))
        self.pixel_std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]))
        # Dummy layers to mimic SAM's structure (replaced actual layers with minimal ops)
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Normalize input (key step causing discrepancy)
        x = (x - self.pixel_mean.view(1, 3, 1, 1)) / self.pixel_std.view(1, 3, 1, 1)
        return self.conv(x)

def my_model_function():
    # Returns a SAM-like model instance with normalization
    return MyModel()

def GetInput():
    # Generate random input matching SAM's expected dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

