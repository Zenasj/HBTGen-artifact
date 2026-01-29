# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assumed input shape (batch, channels, height, width)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN structure inferred for distributed testing context
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 10)  # Derived from 32x32 input size after pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6 * 14 * 14)
        return self.fc1(x)

def my_model_function():
    # Returns a simple CNN instance for distributed testing
    return MyModel()

def GetInput():
    # Returns random tensor matching the inferred input shape
    B = 4  # Arbitrary batch size
    return torch.rand(B, 3, 32, 32, dtype=torch.float32)

