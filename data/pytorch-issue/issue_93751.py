# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

