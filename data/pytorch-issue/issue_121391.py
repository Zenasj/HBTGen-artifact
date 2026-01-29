# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: Batch x Channels x Height x Width
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        # Simulate control flow replacement with torch._assert()
        # (e.g., replacing checks that caused FX tracing issues)
    
    def forward(self, x):
        # Use torch._assert() to replace problematic control flow
        torch._assert(x.ndim == 4, "Input must be 4-dimensional")
        torch._assert(x.shape[1] == 3, "Input must have 3 channels")
        
        x = self.conv(x)
        x = self.bn(x)
        return x

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching expected input shape (B=1, C=3, H=224, W=224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

