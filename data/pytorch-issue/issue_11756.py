# torch.rand(B, 3, 112, 112, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Inferred from input shape and common CNN patterns
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 56 * 56, 10)  # After pooling: 112/2=56
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn(self.conv1(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Initialize with default parameters
    return MyModel()

def GetInput():
    # Generate input matching the model's expected dimensions
    B = 12  # Matches the batch size in original test code
    return torch.rand(B, 3, 112, 112, dtype=torch.float32)

