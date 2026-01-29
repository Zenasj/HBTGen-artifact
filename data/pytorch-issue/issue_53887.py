# torch.rand(2, 2, 1, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inorm1 = nn.InstanceNorm2d(2, affine=False, track_running_stats=True)
        
    def forward(self, x):
        return self.inorm1(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2, 1, 4, dtype=torch.float32)

