# torch.rand(B, 3, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2, dtype=torch.complex64)  # Complex parameter tensor
    
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns a model with complex parameters to test serialization
    return MyModel()

def GetInput():
    # Returns complex input matching model's expected input shape (batch, 3)
    B = 2  # Arbitrary batch size
    return torch.rand(B, 3, dtype=torch.complex64)

