# torch.rand(1, 4, 256, 256, dtype=torch.float32)
import torch
from torch import nn

class Net2_original(nn.Module):
    def __init__(self):
        super().__init__()
        self.subnet = nn.Sequential(nn.Conv2d(4, 4, 1))
        self.net = nn.Sequential(self.subnet)

    def forward(self, x):
        return self.net(x)

class Net2_corrected(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Sequential(nn.Conv2d(4, 4, 1)))

    def forward(self, x):
        return self.net(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_original = Net2_original()
        self.net_corrected = Net2_corrected()

    def forward(self, x):
        output_original = self.net_original(x)
        output_corrected = self.net_corrected(x)
        return torch.tensor([torch.allclose(output_original, output_corrected)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, 256, 256, dtype=torch.float32)

