# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # Matches input dimension

    def forward(self, x):
        # Demonstrates loop with integer constants causing Dynamo specialization
        for i in range(10):
            x = self.linear(x + i)  # Loop variable i treated as constant in each iteration
        return x

def my_model_function():
    # Returns a model instance with random weights
    return MyModel()

def GetInput():
    # Generates batched input tensor matching model's expectation
    B = 1  # Arbitrary batch size
    return torch.rand(B, 10, dtype=torch.float32)

