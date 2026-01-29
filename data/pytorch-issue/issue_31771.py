# torch.rand(B, 1, 32, 32, dtype=torch.float32)  # Matches the input shape of RandomDataset's data
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture matching input shape (1 channel, 32x32)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 15 * 15, 100)  # 32-3=30 → 30/2=15 after pooling
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 10 * 15 * 15)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input
    return torch.rand(5, 1, 32, 32, dtype=torch.float32)  # Batch size 5, 1 channel, 32x32

