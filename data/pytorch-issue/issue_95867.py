# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape for ResNet18

import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, pretrained=False):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, using the pretrained model to avoid the ONNX export issue
    return MyModel(pretrained=True)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

