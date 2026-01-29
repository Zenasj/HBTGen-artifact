# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import os
import torch
import torch.nn as nn

# Workaround for Anaconda DLL loading issue
# Replace 'path/to/torch/lib' with actual path to PyTorch's lib directory
os.environ['PATH'] += os.pathsep + 'path/to/torch/lib'

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 10)  # Example output layer
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

