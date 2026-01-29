# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            groups=1, 
            bias=True
        )

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

