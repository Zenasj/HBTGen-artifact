# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Returns a ResNet18 model instance
    return MyModel()

def GetInput():
    # Returns a random input tensor matching ResNet18's expected input shape
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

