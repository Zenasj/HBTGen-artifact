# torch.rand(4, 5, dtype=torch.float32)
import torch
from torch import nn

class Case1Module(nn.Module):
    def forward(self, x):
        return x.bernoulli(0.2)  # Failing case with p parameter

class Case4Module(nn.Module):
    def forward(self, x):
        return x.bernoulli()      # Working case without parameters

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.case1 = Case1Module()  # Encapsulated failing model
        self.case4 = Case4Module()  # Encapsulated working model

    def forward(self, x):
        # Return outputs of both models for comparison
        # Export will fail due to case1's unsupported operator
        return self.case1(x), self.case4(x)

def my_model_function():
    # Returns the fused model containing both cases
    return MyModel()

def GetInput():
    # Returns input matching the expected shape (4,5)
    return torch.rand(4, 5)

