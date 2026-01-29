# torch.rand(B, C, H, W, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, 5, padding=2, bias=True)
        self.prelu1 = nn.PReLU(64, init=0.25)
        self.conv2 = nn.Conv2d(5, 32, 3, padding=1, bias=True)
        self.prelu2 = nn.PReLU(32, init=0.25)
        self.linear = nn.Linear(64, 1)  # Matches last dimension of cat output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.prelu1(self.conv1(x))
        x2 = self.prelu2(self.conv2(x))
        out = torch.cat((x1, x2), dim=1)  # Concat on channel dimension
        out = self.linear(out)  # Processes last dimension (64 → 1)
        out = self.sigmoid(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 5, 64, 64, dtype=torch.float)  # Matches original input shape

