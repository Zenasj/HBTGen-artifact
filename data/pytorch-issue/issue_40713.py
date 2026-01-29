# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with a dummy Linear layer to avoid Optional issues
        self.submod = nn.Linear(10, 5)  # Default input features = 10

    def init_params(self, input):
        # Dynamically replace the dummy layer with correct dimensions
        in_features = input.size(-1)
        self.submod = nn.Linear(in_features, 5)

    def forward(self, input):
        return self.submod(input)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor matching the dummy Linear's input features (10)
    return torch.rand(2, 10)  # B=2, features=10

