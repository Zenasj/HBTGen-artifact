# torch.rand(1, 3, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        return torch.mean(x, dim=2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 100, 100)

