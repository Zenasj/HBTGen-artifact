# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet50()  # Use ResNet50 as in the issue's example

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

