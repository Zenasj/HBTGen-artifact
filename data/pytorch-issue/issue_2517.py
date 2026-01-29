# torch.rand(64, 10)  # Input shape inferred from the minimal reproducing example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10, 4)  # Matches the nn.Linear(10,4) in the issue's example
        
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    # Returns the model instance. No weights initialization beyond default
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape (64 samples, 10 features)
    return torch.rand(64, 10)  # Matches the Variable(torch.rand(64, 10)) in the example

