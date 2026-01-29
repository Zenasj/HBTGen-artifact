# torch.rand(1, 1, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Matches original LinearModel structure

    def forward(self, x):
        def forward_function(x):
            return self.linear(x)  # Nested function causing guard issue
        return forward_function(x)

def my_model_function():
    return MyModel()  # Returns initialized model instance

def GetInput():
    return torch.rand(1, 1, dtype=torch.float)  # Matches input shape (B=1, features=1)

