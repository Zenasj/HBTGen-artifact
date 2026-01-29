# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, bias=False)
        self.linear1 = nn.Linear(40 * 5 * 5, 300, bias=False)
        self.linear2 = nn.Linear(300, 10, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.pool(self.conv1(input)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size placeholder (can be any positive integer)
    return torch.rand(B, 3, 32, 32, dtype=torch.float32)

