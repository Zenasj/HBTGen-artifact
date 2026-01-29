# torch.rand(B, 3, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Minimal example layer matching input shape from the issue's example
        self.linear = nn.Linear(3, 1)  # 3 input features → 1 output
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns a simple model instance
    return MyModel()

def GetInput():
    # Matches the input shape from the issue's example (2x3 tensor)
    return torch.rand(2, 3, dtype=torch.float)

