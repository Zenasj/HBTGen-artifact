# torch.rand(B, 100, dtype=torch.float32)  # Inferred input shape based on ProceduralDataset example
import torch
import torch.nn as nn
from torch.distributions import Distribution  # Required for ProceduralDataset-like usage

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple model structure to process input tensors from ProceduralDataset-like datasets
        self.layer = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected input shape
    B = 4  # Example batch size
    return torch.rand(B, 100, dtype=torch.float32)

