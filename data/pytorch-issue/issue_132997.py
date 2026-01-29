# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # First BatchNorm2d (initialized with inferred parameters)
        self.bn1 = nn.BatchNorm2d(3)  # Assuming input has 3 channels
        # Conv2d layer (common configuration for demonstration)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # Second BatchNorm2d (output channels from conv1)
        self.bn2 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        return x

def my_model_function():
    # Returns the model instance with default-initialized parameters
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

