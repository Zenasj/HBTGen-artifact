# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn
from collections import namedtuple

MyOutput = namedtuple('MyOutput', ['output'])

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3)
        self.fc = nn.Linear(6 * 30 * 30, 10)  # Assuming input is 32x32 after conv
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return MyOutput(output=x.to('cpu'))  # Explicit CPU handling per fix
        
def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 32, 32, dtype=torch.float32)

