# torch.rand(B, 3, 600, 800, dtype=torch.float32)
from torch import nn
import torch
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.resnet18()  # Uses default pretrained weights (as in original code)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = x.float()  # Matches original code's explicit type conversion
        return self.model(x)

def my_model_function():
    return MyModel()  # Uses default initialization (pretrained ResNet18 with custom FC layer)

def GetInput():
    # Random input matching expected ResNet format (B,C,H,W) with C=3
    return torch.rand(1, 3, 600, 800, dtype=torch.float32)

