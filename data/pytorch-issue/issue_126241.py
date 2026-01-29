# torch.rand(4, 3, 224, 224, dtype=torch.bfloat16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, dtype=torch.bfloat16)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False, dtype=torch.bfloat16)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def my_model_function():
    model = MyModel()
    model.to('cuda')
    return model

def GetInput():
    return torch.randn(4, 3, 224, 224, dtype=torch.bfloat16, device='cuda')

