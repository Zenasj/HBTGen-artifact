# torch.rand(B, 30, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 5)
        self.fc2 = nn.Linear(5, 3)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def my_model_function():
    torch.manual_seed(1234)  # Matches original weight initialization seed
    return MyModel()

def GetInput():
    return torch.rand(3, 30, dtype=torch.float32)

