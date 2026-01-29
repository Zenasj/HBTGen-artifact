# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 10, dtype=torch.float32)

