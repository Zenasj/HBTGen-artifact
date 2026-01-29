# torch.rand(B, 10)  # Assuming input is a batch of 1D tensors with 10 features
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple model to process input tensors from the dataset
        self.fc = nn.Linear(10, 5)  # Example layer with input size 10

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape
    B = 4  # Arbitrary batch size
    return torch.rand(B, 10)  # Matches the input shape comment above

