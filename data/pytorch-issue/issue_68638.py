# torch.rand(1, 12, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3)
        self.conv_1 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
        
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 12, 64, 64, dtype=torch.float32)

