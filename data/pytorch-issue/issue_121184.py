# torch.rand(1, 3, 64, 64, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        t1 = self.conv(x)
        self.bn.eval()
        self.bn.running_mean.zero_()
        self.bn.running_var.fill_(1)
        return self.bn(t1)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3, 64, 64, dtype=torch.float32)

