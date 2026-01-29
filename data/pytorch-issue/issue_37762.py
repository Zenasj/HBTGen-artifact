# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # Simplest possible model for demonstration

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns a minimal model with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the model's input requirements
    return torch.rand(1, 10, dtype=torch.float32)

