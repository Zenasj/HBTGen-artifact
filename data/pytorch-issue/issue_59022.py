# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Example input shape from CI test scenarios
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Problematic __import__ usage from the example
        self.fake_mod = __import__("time")  # Triggers packaging error/warning
        # Dummy layers to form a valid model
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns model instance with problematic __import__ in initialization
    return MyModel()

def GetInput():
    # Matches expected input shape for the model's convolution layer
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

