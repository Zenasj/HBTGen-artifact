# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet50()  # Base model from the issue's reproduction code

    def forward(self, x):
        return self.model(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Matches the original code's evaluation mode
    return model

def GetInput():
    # Generates a random input matching the expected shape for ResNet50 (3-channel, 224x224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

