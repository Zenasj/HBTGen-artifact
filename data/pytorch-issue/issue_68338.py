# torch.rand(B, C, H, W, dtype=torch.float32)  # Assuming input shape (B=1, C=3, H=32, W=32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16*32*32, 10)  # Example output size
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

