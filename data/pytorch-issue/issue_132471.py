import torch
import torch.nn as nn

# torch.rand(B, 100, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(100, 100))

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size, can be any positive integer
    return torch.rand(B, 100, dtype=torch.float32)

