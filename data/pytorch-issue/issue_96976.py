# torch.rand(1, 3, 64, 64, dtype=torch.float32)
import torch
from torchvision.models import resnet50
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50()  # Use torchvision's ResNet50 as the core model

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Returns the wrapped ResNet50 model
    return MyModel()

def GetInput():
    # Generates a random input tensor matching ResNet's expected dimensions
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

