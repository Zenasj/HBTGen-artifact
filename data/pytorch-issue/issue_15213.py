# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape for image classification
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 10)  # 56x56 from 224 after two pools

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc1(x)
        return x

def my_model_function():
    # Returns a simple CNN model for image classification
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

