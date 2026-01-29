import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# torch.rand(B, 3, 224, 224, dtype=torch.float32)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x, rate):
        # Example operation using the rate parameter (e.g., scaling)
        return self.conv(x) * rate

class MyModel(nn.Module):
    def __init__(self, drop_connect_rate=0.2):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.block = Block()

    def forward(self, x):
        # Workaround using lambda to capture non-tensor arguments for checkpointing
        return checkpoint(lambda x: self.block(x, self.drop_connect_rate), x)

def my_model_function():
    # Returns a model instance with default drop_connect_rate=0.2
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape (B, C, H, W)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

