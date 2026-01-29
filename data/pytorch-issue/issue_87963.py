# torch.rand(B, 3)  # Input shape inferred from nn.Linear(3,4) in the issue's example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(3, 4)  # Matches the Linear layer from the issue's example

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()  # Returns an instance with the problematic Linear layer

def GetInput():
    # Returns a random tensor matching the input expected by MyModel (batch_size=1, features=3)
    return torch.rand(1, 3)

