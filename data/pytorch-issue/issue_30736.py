# torch.rand(B, 1, 64, 64, dtype=torch.float32)  # Assumed input shape based on dataset's processing
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Basic 2D CNN architecture for segmentation (assumed task from dataset)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=1)  # Output channels = 2 (binary segmentation)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        return x

def my_model_function():
    # Returns initialized model instance
    return MyModel()

def GetInput():
    # Generates random input matching expected shape (B=1, C=1, H=64, W=64)
    return torch.rand(1, 1, 64, 64, dtype=torch.float32)

