# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = resnet18(weights=None)  # Matches model initialization in the issue

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

