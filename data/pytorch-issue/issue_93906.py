# torch.rand(4, 3, 224, 224, dtype=torch.float32)
import torch
from torchvision.models import swin_v2_s
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = swin_v2_s()  # Use torchvision's SwinV2 model

    def forward(self, x):
        return self.model(x)  # Forward pass through the model

def my_model_function():
    return MyModel()  # Return the wrapped SwinV2 model

def GetInput():
    return torch.randn(4, 3, 224, 224)  # Batch of images with standard input shape

