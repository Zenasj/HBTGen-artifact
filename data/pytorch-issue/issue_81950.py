# torch.rand(B, 100, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.complex64)
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=torch.complex64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Example batch size
    return torch.rand(B, 100, dtype=torch.complex64)

