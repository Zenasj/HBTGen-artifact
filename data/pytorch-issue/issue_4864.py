# torch.rand(143, 3, 224, 224, dtype=torch.float32)  # Inferred input shape from last batch size
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18()  # Standard ResNet-18 architecture

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    # Returns initialized ResNet-18 model with default weights
    return MyModel()

def GetInput():
    # Returns a random tensor matching the problematic last batch size (143)
    return torch.rand(143, 3, 224, 224, dtype=torch.float32, device='cuda')

