# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(2, 10)  # Matches the MLPModule structure from the test

    def forward(self, x):
        return self.net1(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generates input matching the model's expected input shape (batch_size, 2)
    return torch.rand(4, 2, dtype=torch.float32)  # Arbitrary batch size 4 for demonstration

