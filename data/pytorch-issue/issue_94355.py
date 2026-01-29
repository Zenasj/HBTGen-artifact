# torch.rand(B, 10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(20, 10)  # Matches input shape (batch, 10 features) from example

