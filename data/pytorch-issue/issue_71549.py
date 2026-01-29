# torch.rand(1, 3, 2, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.AvgPool2d(
            kernel_size=(3, 3),
            stride=(3, 3),
            padding=(1, 1),
            ceil_mode=True
        )
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 2, 2, dtype=torch.float32)

