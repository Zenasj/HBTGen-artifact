# torch.randint(0, 10, (3,))  # Inferred input shape (integer tensor of shape (3,))
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, size, maxint=10):
        super().__init__()
        self.maxint = maxint
        self.weight = torch.randint(0, maxint, [size])  # Matches TestMod's initialization

    def forward(self, x):
        # Forward uses self.weight (from TestMod's implementation)
        return F.one_hot(self.weight, num_classes=self.maxint)

def my_model_function():
    # Replicates TestMod(3) from the original example
    return MyModel(3)

def GetInput():
    # Returns a tensor matching the shape used in the original export example (test.weight)
    return torch.randint(0, 10, (3,))

