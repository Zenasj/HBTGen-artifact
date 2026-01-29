# torch.rand(B, 3, 32, 32, dtype=torch.float)  # Assuming image input with 3 channels and 32x32 resolution
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 10)  # Example 10-class classifier

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()  # Default initialization suffices for demonstration

def GetInput():
    return torch.rand(4, 3, 32, 32, dtype=torch.float)  # Batch size 4, 3-channel 32x32 images

