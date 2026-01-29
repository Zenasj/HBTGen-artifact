# torch.rand(B, 1, 224, 224, dtype=torch.float32)  # Inferred input shape (batch, channels, height, width)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Example layer requiring CUDA
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Force CUDA operation to trigger crash on non-NVIDIA systems with CUDA build
        x = x.cuda()  # This line will crash if CUDA is unavailable
        x = self.pool(F.relu(self.conv(x)))
        return x

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random CPU tensor matching the model's input requirements
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

