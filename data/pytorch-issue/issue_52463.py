# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Input shape for MobileNetV3
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.mobilenet_v3_large(pretrained=False)  # Matches torchvision's default

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Returns initialized MobileNetV3 model (non-pretrained)
    return MyModel()

def GetInput():
    # Returns a random input tensor matching MobileNetV3's expected input
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

