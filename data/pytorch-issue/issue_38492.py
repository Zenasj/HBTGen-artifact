# torch.rand(B, 3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 10)  # Matches input shape from dataset's "foo" tensor
    
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size (matches original issue's batch_size=2)
    return torch.rand(B, 3, dtype=torch.float)

