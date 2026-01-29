# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # ResNet18 input shape (batch, channels, height, width)
import torch
import torch.nn as nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = resnet18()  # Base model from torchvision

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    # Initialize the model with default weights
    model = MyModel()
    return model

def GetInput():
    # Generate random input tensor matching ResNet18 requirements
    batch_size = 3  # Example batch size from issue logs
    return torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)

