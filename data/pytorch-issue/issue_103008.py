import torch
from torch import nn
from collections import namedtuple

# torch.rand(B, 3, H, W, dtype=torch.float32)  # e.g., (1, 3, 224, 224)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # Define a namedtuple with default None for 'processed' to trigger Dynamo error
        self.Token = namedtuple('Token', ['data', 'processed'], defaults=(None,))

    def forward(self, x):
        features = self.conv(x)
        # Create Token with default value (processed=None) to cause assertion failure
        token = self.Token(data=features, processed=None)
        return token.data.sum()  # Use the data field to return a tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Returns random tensor matching input shape (B, C, H, W)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

