# torch.rand(B, 3, 416, 416, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified YOLOv3-like architecture based on common structures
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def my_model_function():
    # Initialize model with default parameters
    model = MyModel()
    return model

def GetInput():
    # Generate random input matching YOLOv3 input dimensions (batch=2 for multiprocessing context)
    return torch.rand(2, 3, 416, 416, dtype=torch.float)

