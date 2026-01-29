# torch.rand(1, 3, 3200, 3200, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5, device=torch.device('cuda'))
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 3200, 3200, dtype=torch.float32, device=torch.device('cuda'))

