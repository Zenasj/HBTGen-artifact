# torch.rand(B, 10, dtype=torch.float)  # Input shape: batch_size x 10
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('my_buffer', None)
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        if self.my_buffer is None:
            # Initialize buffer dynamically during first forward pass
            self.my_buffer = torch.randn(5).to(x.device)
        return self.linear(x) + self.my_buffer

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor compatible with MyModel
    return torch.rand(1, 10, dtype=torch.float)

