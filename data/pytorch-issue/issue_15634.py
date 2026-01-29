# torch.randint(-10, 10, (4, 4), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(4, 4) * 10).long()

