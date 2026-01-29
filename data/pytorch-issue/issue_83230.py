# torch.rand(1, 2, 3, 6, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(2, 2, stride=1, kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def my_model_function():
    model = MyModel()
    # Ensure batchnorm is in eval mode as per issue's setup
    model.eval()
    return model

def GetInput():
    return torch.rand(1, 2, 3, 6, dtype=torch.float32)

