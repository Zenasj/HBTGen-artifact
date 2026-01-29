# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Inferred input shape for a typical CNN
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simulating a 2D CNN architecture commonly used in detection/segmentation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)  # After pooling (224/2/2=56)
        self.fc2 = nn.Linear(256, 10)  # Output layer placeholder

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns an instance of the model with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected shape
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)  # Batch size 2, 3 channels, 224x224

