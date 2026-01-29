# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assumed input shape for a typical CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture to demonstrate device/backend compatibility
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)  # Adjusted for MaxPool downsampling

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        return self.fc(x)

def my_model_function():
    # Returns a model instance with dummy initialization
    model = MyModel()
    return model

def GetInput():
    # Generates a random input tensor matching the expected shape
    return torch.rand(4, 3, 32, 32, dtype=torch.float32)  # Batch size 4, 3 channels, 32x32

