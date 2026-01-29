import torch
import torch.nn as nn

# torch.rand(B, 3, 32, 32, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2)  # Not used in forward, as per original code
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        res1 = self.relu(self.conv2(x1) + self.conv3(x1))
        res2 = self.relu2(self.conv4(res1) + res1)
        return res2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

