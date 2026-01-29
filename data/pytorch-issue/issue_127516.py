# torch.rand(4, 4, dtype=torch.float32)  # Inferred input shape from the issue's example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, add_fn):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.add_fn = add_fn
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(4, 4)
        self.linear4 = nn.Linear(4, 4)
        self.add_fn2 = add_fn
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        tmp = self.add_fn(x1, x2)
        # The following lines were commented in the original issue's code
        # tmp1 = self.linear3(tmp)
        # tmp2 = self.linear4(tmp)
        # res = self.add_fn2(tmp1, tmp2)
        # return res
        return tmp

def my_model_function():
    add_fn = lambda x, y: x.add_(y)
    return MyModel(add_fn)

def GetInput():
    return torch.randn((4, 4), dtype=torch.float32, requires_grad=False).add(1)

