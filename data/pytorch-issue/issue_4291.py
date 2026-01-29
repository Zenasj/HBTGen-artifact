# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
        # Initialize weights to large values to simulate potential overflow scenarios
        self.fc1.weight.data.fill_(1000.0)
        self.fc2.weight.data.fill_(1000.0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape expected by MyModel (batch_size=128, features=10)
    return torch.rand(128, 10, dtype=torch.float32)

