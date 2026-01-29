# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x).mean()  # Returns scalar for backward()

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a 4D tensor matching the model's expected input
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

