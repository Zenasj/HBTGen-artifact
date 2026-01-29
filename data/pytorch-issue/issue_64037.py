# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Example input shape for a simple CNN
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)  # Example output for 10-class classification

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns a simple CNN model instance
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

