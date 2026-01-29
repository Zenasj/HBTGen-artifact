# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # MobileNetV2 input shape
import torch
import torchvision
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

