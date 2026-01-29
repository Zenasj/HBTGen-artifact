# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: [1, 3, 224, 224]
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()  # Matches the issue's evaluation mode

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

