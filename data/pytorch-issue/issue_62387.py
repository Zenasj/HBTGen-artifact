# torch.rand(5, 3, 224, 224, dtype=torch.float32)

import torch
import torchvision.models as models
from torch.nn import Module

class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 3, 224, 224, dtype=torch.float32)

