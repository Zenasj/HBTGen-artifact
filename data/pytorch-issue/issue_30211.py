# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    
    def forward(self, x):
        return self.conv_transpose(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 128, 128, 80, dtype=torch.float32)

