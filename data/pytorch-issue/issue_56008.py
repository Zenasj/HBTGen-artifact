import torch
import torch.nn as nn
from torchvision import models

# torch.rand(B, 3, 224, 224, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(num_classes=10)
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

