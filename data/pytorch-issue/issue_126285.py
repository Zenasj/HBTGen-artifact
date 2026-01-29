# torch.rand(B, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.rand(2, 2))
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        
    def forward(self, x):
        # Forward path inferred from model structure (layer1 -> layer2)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def my_model_function():
    # Returns model instance with default initialization
    return MyModel()

def GetInput():
    # Generates input tensor matching expected shape (B, 4)
    B = 2  # Example batch size (can be adjusted)
    return torch.rand(B, 4, dtype=torch.float32)

