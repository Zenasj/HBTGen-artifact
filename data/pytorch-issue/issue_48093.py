# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.shufflenet_v2_x0_5(pretrained=True)
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 6  # Batch size that triggered the error in the issue
    return torch.randn(B, 3, 224, 224, dtype=torch.float32)

