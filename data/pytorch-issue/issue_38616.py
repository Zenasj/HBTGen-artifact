# torch.rand(B, C, H, W, dtype=torch.float32)  # Example: (1, 2, 1, 1)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer to process input flattened from (C,H,W)
        self.fc = nn.Linear(2, 1)  # Input size=2 (from 2x1x1), output size=1
        
    def forward(self, x):
        # Flatten input tensor for linear layer
        return self.fc(x.view(x.size(0), -1))

def my_model_function():
    # Returns a simple model that processes 2-element inputs
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected shape (B=1, C=2, H=1, W=1)
    return torch.rand(1, 2, 1, 1, dtype=torch.float32)

