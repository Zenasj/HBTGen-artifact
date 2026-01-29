# torch.rand(32, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18()

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 3, 224, 224, dtype=torch.float32)

