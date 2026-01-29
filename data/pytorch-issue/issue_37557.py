# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Simulates the torch.add operation used in the test case
        return torch.add(x, x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the 1-element tensor used in the test (torch.ones(1))
    return torch.rand(1)

