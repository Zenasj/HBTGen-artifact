# torch.rand(B, 10, dtype=torch.float32)  # Input shape inferred from the model's Linear layers (input features=10)
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    # Returns the base model before DDP wrapping (as in the issue's original code)
    return MyModel()

def GetInput():
    # Random input matching the model's expected input (batch_size=2 chosen to align with the issue's multi-GPU setup)
    return torch.rand(2, 10, dtype=torch.float32)

