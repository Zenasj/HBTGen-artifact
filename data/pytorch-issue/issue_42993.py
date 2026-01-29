# torch.rand(B, 1, H, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.n = nn.Conv2d(1, 35, 1)

    def forward(self, x):
        x = self.n(x)
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.flatten(3, 4)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 1024, 2)

