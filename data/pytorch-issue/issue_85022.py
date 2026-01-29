# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16_bn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = vgg16_bn(pretrained=True)  # Uses the older pretrained flag approach

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Returns the VGG16_BN model initialized with pretrained weights (as per older torchvision versions)
    return MyModel()

def GetInput():
    # Returns a random input tensor matching VGG's expected input shape (B, 3, 224, 224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

