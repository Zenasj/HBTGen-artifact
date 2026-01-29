# torch.rand(B, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class SomeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.rand(4, 4))

    def forward(self, x):
        return torch.mm(x, self.param)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_mods = 3
        # Fix: Create distinct instances instead of shallow copies
        self.modlist = nn.ModuleList([SomeModule() for _ in range(self.num_mods)])

    def forward(self, x):
        for i in range(self.num_mods):
            x = self.modlist[i](x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 4, dtype=torch.float32)

