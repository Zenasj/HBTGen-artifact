# torch.rand(2, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)  # Matches the reduced repro's BatchNorm module

    def forward(self, x):
        return self.bn(x)  # Forward pass triggers BatchNorm's in-place buffer updates

def my_model_function():
    model = MyModel()
    model.train()  # Error occurs only in training mode (as noted in comments)
    return model

def GetInput():
    return torch.randn(2, 3, 224, 224)  # Matches the reduced repro's input shape

