# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape for image classification
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 224 * 224, 10)  # Dummy output layer for classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns a simple CNN model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected model input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

