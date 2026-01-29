# torch.rand(B, 784, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden1 = 512
        self.hidden2 = 512
        self.fc1 = nn.Linear(28 * 28, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 28 * 28, dtype=torch.float32)

