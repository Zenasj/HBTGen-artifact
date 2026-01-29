# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Reproduces the scenario where Adamax's stride issue occurs
        self.linear = nn.Linear(3, 2)  # Matches the 2x3 tensor example from the issue

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(1, 3, dtype=torch.float32)  # Matches the model's input dimension (batch=1, features=3)

