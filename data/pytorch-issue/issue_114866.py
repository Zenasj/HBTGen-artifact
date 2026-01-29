# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.a = nn.Parameter(torch.tensor(12.0, dtype=torch.float32), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        c = torch.tensor(0.0, dtype=self.a.dtype, device=self.a.device)
        c += torch.sqrt(self.a)
        c += torch.cos(self.a)
        c += torch.cosh(self.a)
        c += torch.sin(self.a)
        c += torch.sinh(self.a)
        c += torch.tan(self.a)
        c += torch.tanh(self.a)
        c += torch.asin(self.b)
        c += torch.acos(self.b)
        c += torch.atan(self.a)
        return x + c

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32)

