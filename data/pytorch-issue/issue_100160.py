# torch.rand(1, 32, 8, 8, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(32, 10, 6, padding='same', padding_mode="circular")
        self.linear = nn.Linear(640, 6)  # 10*8*8 = 640 features after flatten

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 32, 8, 8, dtype=torch.float32)

