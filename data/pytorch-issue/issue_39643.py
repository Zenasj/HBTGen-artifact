# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Bone module (backbone with BatchNorm)
        self.bone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Head module (classification head with BatchNorm)
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Forward through bone and head modules
        x = self.bone(x)
        return self.head(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching model's expected input shape
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

