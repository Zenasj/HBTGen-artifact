# torch.rand(B, 10, dtype=torch.float32, requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, dtype=torch.float32, requires_grad=True)

