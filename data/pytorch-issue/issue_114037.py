# torch.rand(2, 3, 8, 8, dtype=torch.float32)  # Input shape inferred from original code
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(3)  # Matches input's channel dimension (3)
    
    def forward(self, x):
        x = self.batchnorm2d(x)
        x_ = torch.linalg.slogdet(x)  # Retained as per original code logic
        x_erfinv = torch.special.erfinv(x)  # Potential source of NaNs due to input range
        return x_erfinv

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input matching the expected shape (2, 3, 8, 8)
    return torch.rand(2, 3, 8, 8)

