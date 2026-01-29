# torch.rand(1, 4, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Reproduces the Conv2d configuration from the original issue
        self.conv = nn.Conv2d(4, 50, kernel_size=10, stride=5)
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random input matching the expected shape (B=1, C=4, H=100, W=100)
    return torch.rand(1, 4, 100, 100, dtype=torch.float32)

