# torch.rand(8192, 3072, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3 * 1024, 2 * 1024, bias=False)
        self.lin2 = nn.Linear(2 * 1024, 2 * 1024, bias=False)
        self.lin3 = nn.Linear(2 * 1024, 3 * 1024, bias=False)  # Concatenates along dim 0 by default

    def forward(self, x):
        t1 = self.lin1(x).relu().add(1.0)
        t2 = self.lin2(t1).relu().pow(2.0)
        c1 = torch.cat((t1, t2))  # Default dim=0 creates (batch*2, 2048)
        t3 = self.lin3(c1).relu().mul(2.0)
        return t3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8 * 1024, 3 * 1024, dtype=torch.float32)

