# torch.rand(1, 1, 64, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features=1)
        self.conv = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3)
        self.activation = nn.ReLU(inplace=True)
        self.final = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        y = self.norm(x)
        y = self.conv(y)
        y = self.activation(y)
        y = self.final(y)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 64, 64, 64, dtype=torch.float32)

