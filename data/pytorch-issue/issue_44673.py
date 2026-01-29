# torch.rand(B, 10, dtype=torch.float32)  # Example input shape from DDP training (B=20)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)  # Matches the DDP example's input/output dimensions

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    # Returns a model instance matching the DDP example's structure
    return MyModel()

def GetInput():
    # Returns input matching the DDP example's torch.randn(20, 10)
    return torch.randn(20, 10, dtype=torch.float32)

