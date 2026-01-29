# torch.rand(B, 1, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Basic CNN for MNIST classification (inferred from dataset usage)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc = nn.Linear(32 * 26 * 26, 10)  # 10-class output for MNIST
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Returns a simple MNIST classifier model
    return MyModel()

def GetInput():
    # Matches MNIST input shape (batch, channels, height, width)
    B = 4  # Batch size from issue's DataLoader setup
    return torch.rand(B, 1, 28, 28, dtype=torch.float32)

