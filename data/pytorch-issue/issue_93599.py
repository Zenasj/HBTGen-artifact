# torch.rand(4096, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    model = MyModel()
    model.eval()
    model.to('cuda')
    return model

def GetInput():
    return torch.randn((4096, 3, 224, 224), device='cuda', dtype=torch.float32)

