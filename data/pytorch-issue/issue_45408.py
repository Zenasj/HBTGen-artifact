# torch.rand(128, 3, 100, 100, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet101(pretrained=True)

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(128, 3, 100, 100, dtype=torch.float32)

