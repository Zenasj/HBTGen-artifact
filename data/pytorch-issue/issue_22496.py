# torch.rand(1, 1, 256, 256, 256, dtype=torch.float32).cuda()  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.layer_2 = nn.Conv3d(1, 1, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 256, 256, 256, dtype=torch.float32).cuda()

