# torch.rand(B, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(1, 1)  # Simple linear layer for demonstration

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor with shape (batch_size, 1)
    # Using batch size 1 as in the original DataLoader example
    return torch.rand(1, 1, dtype=torch.float32)

