# torch.rand(1, 1, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class AddModule(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, x):
        return x + self.value

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.add1 = AddModule(1.0)  # Represents the first remote add operation (value=1)
        self.add3 = AddModule(3.0)  # Represents the second remote add operation (value=3)

    def forward(self, x):
        # Simulates combining results from two operations (similar to the issue's example)
        a = self.add1(x)
        b = self.add3(x)
        return a + b  # Merged output as in the issue's "x = rref1.to_here() + rref2.to_here()"

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input matching the expected shape (B=1, C=1, H=2, W=2)
    return torch.rand(1, 1, 2, 2, dtype=torch.float32)

