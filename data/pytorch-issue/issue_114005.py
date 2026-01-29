# torch.rand(B, C, H, W, dtype=torch.float32)  # Example input shape: (1, 3, 224, 224)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 56 * 56, 10)  # 56x56 after pooling
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return x

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

