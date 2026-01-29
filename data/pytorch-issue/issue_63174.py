# torch.rand(B, 3, 32, 32, dtype=torch.float)  # Example input shape for a CNN
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_model_function():
    # Returns a simple CNN model instance
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected dimensions
    batch_size = 4  # Example batch size
    return torch.rand(batch_size, 3, 32, 32, dtype=torch.float)

