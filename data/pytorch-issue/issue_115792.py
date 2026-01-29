# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimics ResNet18's first convolution layer structure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        return self.conv1(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor matching ResNet18's expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

