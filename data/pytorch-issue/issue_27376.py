import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    class ScaleFactorModel(nn.Module):
        def forward(self, x):
            return F.interpolate(x, mode='nearest', scale_factor=2)
    
    class SizeModel(nn.Module):
        def forward(self, x):
            return F.interpolate(x, mode='nearest', size=200)
    
    def __init__(self):
        super().__init__()
        self.model1 = self.ScaleFactorModel()
        self.model2 = self.SizeModel()
    
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return (out1, out2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 100, 100)

