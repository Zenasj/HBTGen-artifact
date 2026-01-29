# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assuming 3-channel images of size 32x32
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)  # Assuming 10-class classification

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching expected shape (B, C, H, W)
    return torch.rand(4, 3, 32, 32, dtype=torch.float32)  # Batch size 4 as an example

