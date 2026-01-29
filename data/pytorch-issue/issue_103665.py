import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Problematic configuration (209 channels causes the error)
        self.conv = nn.Conv2d(209, 209, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 209, 2, 2, dtype=torch.float32).cuda()

