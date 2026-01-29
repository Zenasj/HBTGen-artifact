# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Inferred input shape: batch x channels x height x width
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16*32*32, 10)  # 32x32 image after conv layer

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Return a simple CNN model instance
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (B=1, C=3, H=32, W=32)
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

