# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape for a typical CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 10)  # Example output for classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Batch size (arbitrary choice)
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

