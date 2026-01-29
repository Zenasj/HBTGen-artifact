# torch.rand(B, 3, 800, 800, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 32, 1)
        self.final_conv = nn.Conv2d(32, 20, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.final_conv(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Matches per-GPU batch size (original setup: 8 GPUs with total batch_size 8)
    return torch.rand(B, 3, 800, 800, dtype=torch.float)

