# torch.rand(4, 4, dtype=torch.float32, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=4, out_features=16, bias=False).cuda()

    def forward(self, x):
        x = self.lin(x)
        x = x.view(4, 4, -1)
        x = x.transpose(-1, -2)
        x /= 2.0
        return (x,)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, device="cuda")

