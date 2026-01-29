# torch.rand(B, 3, 32, 32, dtype=torch.float)  # Inferred input shape for a common image classification task
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)  # 10-class classifier

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape (B, 3, 32, 32)
    B = 4  # Arbitrary batch size for demonstration
    return torch.rand(B, 3, 32, 32, dtype=torch.float)

