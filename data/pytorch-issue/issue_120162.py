# torch.rand(5, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()  # Base model from torchvision
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()  # Returns initialized ResNet18 model

def GetInput():
    return torch.randn(5, 3, 224, 224, dtype=torch.float32)  # Matches ResNet input requirements

