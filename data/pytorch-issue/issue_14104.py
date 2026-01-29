# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape from common image processing tasks
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x))

def my_model_function():
    return MyModel()  # Basic model initialization

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Matches expected input shape/dtype

