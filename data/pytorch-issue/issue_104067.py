# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.l1 = nn.Linear(2, 2)
        
    def forward(self, x):
        return self.l1(x)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.a = A()
        
    def forward(self, x):
        def custom():
            def custom_forward(x_):
                return self.a(x_)
            return custom_forward
        z = self.l1(checkpoint(custom(), x))
        return z

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2)

