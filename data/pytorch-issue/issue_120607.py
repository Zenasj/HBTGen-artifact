# torch.rand(1, dtype=torch.float32)  # Inferred input shape based on example's scalar-like behavior
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        s = "Hello" + str(x[0].item())  # Mimics string concatenation in original MWE
        print(s)
        return x  # Returns input tensor to maintain model structure

def my_model_function():
    return MyModel()  # Returns the model instance

def GetInput():
    return torch.tensor([0.0])  # Matches the expected input shape (scalar tensor)

