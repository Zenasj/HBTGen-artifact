# torch.rand(B, 1, 32, 32, dtype=torch.float)  # Inferred input shape: grayscale images of size 32x32
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)  # Assuming 2x max pooling layers reduce spatial dims to 8x8

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 1, 32, 32, dtype=torch.float)  # Batch size 4, 1 channel, 32x32 image

