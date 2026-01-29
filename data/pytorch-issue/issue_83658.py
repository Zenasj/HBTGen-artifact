# torch.rand(B, 16, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_feature = 16
        self.oh = 4
        self.ow = 8
        self.out_feature = self.oh * self.ow
        self.linear = torch.nn.Linear(self.in_feature, self.out_feature)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.linear(x)
        y = self.relu(y)
        return y.view(y.size(0), 1, self.oh, self.ow)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 16)

