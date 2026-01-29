# torch.rand(B=5, C=3, H=224, W=224, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50()  # Base model from torchvision

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    # Return initialized ResNet50 model
    return MyModel()

def GetInput():
    # Return CUDA tensor matching ResNet50 input requirements
    return torch.randn(5, 3, 224, 224, dtype=torch.float32).cuda()

