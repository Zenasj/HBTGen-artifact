# torch.rand(B, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=(3, 3))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 84, 84, dtype=torch.float32).cuda()

