import torch
import torchvision.models as models
from torch import nn

# torch.randn(2, 3, 256, 288, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)

    def forward(self, x):
        return self.resnet50(x)

def my_model_function():
    # Returns a ResNet50 model with pretrained weights
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the ResNet50's expected input shape
    return torch.randn(2, 3, 256, 288, dtype=torch.float32)

