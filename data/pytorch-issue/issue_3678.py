# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture for demonstration
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 10)  # Matches input shape (224x224)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)
        return x

def my_model_function():
    # Returns initialized model instance with random weights
    return MyModel()

def GetInput():
    # Returns random tensor matching model's expected input
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

