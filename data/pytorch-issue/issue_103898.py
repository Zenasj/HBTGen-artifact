# torch.rand(B, 3, 640, 1000, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fcn = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
        self.fcn.eval()  # Matches original code's evaluation mode

    def forward(self, x):
        return self.fcn(x)['out']  # Output matches the 'out' key in original usage

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape from trace and issue's example (1,3,640,1000)
    return torch.rand(1, 3, 640, 1000, dtype=torch.float32)

