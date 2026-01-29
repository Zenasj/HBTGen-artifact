# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.m = nn.AdaptiveMaxPool3d((1, None, None))  # preserves depth dimension via None
    
    def forward(self, x):
        return self.m(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 64, 10, 9, 8, dtype=torch.float32)  # matches original dummy input dimensions

