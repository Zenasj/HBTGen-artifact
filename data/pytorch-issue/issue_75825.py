import torch
from torch import nn

# torch.rand(B, 3, 10, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv1d(3, 3, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 10, dtype=torch.float32)

