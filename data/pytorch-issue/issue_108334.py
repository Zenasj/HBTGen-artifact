# torch.rand(1, 3, H, W, dtype=torch.float32)
import random
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=2000, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Matches the reported environment using GPU (A10)
    return model

def GetInput():
    h = random.randint(5, 800)
    w = random.randint(5, 800)
    return torch.randn(1, 3, h, w, dtype=torch.float32).cuda()  # Dynamic H/W as in issue's simplified example

