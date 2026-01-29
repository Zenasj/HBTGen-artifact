# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Inferred input shape based on common image dimensions
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture to trigger GPU code execution
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 10)  # Matches input shape after pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random tensor matching the expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32).cuda()  # Explicit device placement for ROCm testing

