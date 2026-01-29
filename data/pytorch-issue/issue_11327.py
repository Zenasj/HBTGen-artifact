# torch.rand(B, C, H, W, dtype=...)  # (64, 3, 128, 128)  # Inferred input shape based on the provided code

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_size, grid_s, num_classes):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.grid_s = grid_s
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(64 * (input_size[2] // 4) * (input_size[3] // 4), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    input_size = (64, 3, 128, 128)
    grid_s = 8
    num_classes = 9092
    model = MyModel(input_size, grid_s, num_classes)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 64, 3, 128, 128
    input_tensor = torch.rand(B, C, H, W, dtype=torch.float32)
    return input_tensor

