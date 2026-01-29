# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class OrigModel(nn.Module):
    def forward(self, x):
        return x + 1  # Matches the original_graph's add(x, 1)

class PatternModel(nn.Module):
    def forward(self, x):
        a = x * 2  # Creates a non-literal tensor 'a' (key for comparison with literal in orig)
        return x + a  # Matches the pattern_graph's structure (ignoring literals leads to false match)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.orig = OrigModel()  # Original model with literal
        self.pattern = PatternModel()  # Pattern model with non-literal

    def forward(self, x):
        # Returns outputs of both models for comparison
        return self.orig(x), self.pattern(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)  # Scalar input matching both models

