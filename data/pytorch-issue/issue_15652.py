# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assumed input shape: Bx3x32x32 (e.g., RGB images)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # After two conv/pool steps: 32 -> 14 -> 5 (approximate)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_model_function():
    # Returns a base model instance (not wrapped in DataParallel here; user may apply it externally)
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

