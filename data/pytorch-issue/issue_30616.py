# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 112 * 112, 100)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns a model initialized on CPU to allow proper DataParallel handling
    return MyModel()

def GetInput():
    B = 4  # Example batch size
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

