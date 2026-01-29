# torch.rand(B, C, H, dtype=torch.float32)  # Actual input shape: (2, 128, 64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)  # Matches the TestMod structure from the issue
        
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns model instance matching the distributed setup in the issue (ColwiseParallel not implemented here for standalone code)
    return MyModel()

def GetInput():
    # Returns input tensor matching the example's (2, 128, 64) shape
    return torch.randn(2, 128, 64, dtype=torch.float32)

