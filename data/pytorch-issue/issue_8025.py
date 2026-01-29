# torch.rand(B, 1000, dtype=torch.float32)  # Input shape inferred as (batch_size, 1000)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(1000, 100)  # Matches hidden dim H=100
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)    # Matches output dim D_out=10

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def my_model_function():
    # Returns initialized model instance with default weights
    return MyModel()

def GetInput():
    # Returns random input tensor matching expected shape (B, 1000)
    B = 64  # Batch size from issue context
    return torch.rand(B, 1000, dtype=torch.float32)

