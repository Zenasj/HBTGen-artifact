# torch.rand(B, 100, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, c=100):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.bn(x)
        out = torch.nn.functional.relu(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 100, 100, 100, dtype=torch.float32, requires_grad=True)

