# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3, 10)  # Matches input dimensionality

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns a simple linear model instance
    return MyModel()

def GetInput():
    # Generates a random tensor matching the model's expected input shape
    return torch.rand(1, 3, dtype=torch.float32)

