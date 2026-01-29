# torch.rand(B, 8, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def my_model_function():
    # Initialize model on CUDA as in the original example
    return MyModel().cuda()

def GetInput():
    # Generate input with batch size 4 (matches original example)
    return torch.randn(4, 8).cuda()

