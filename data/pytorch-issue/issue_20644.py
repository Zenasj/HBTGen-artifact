# torch.rand(B, 64, H, W, dtype=torch.float32)  # e.g., (1, 64, 32, 32)
import torch
import torch.nn as nn
from torch.jit import ScriptModule

class MultiBox(ScriptModule):
    def __init__(self):
        super().__init__()
        loc_layers = nn.ModuleList()
        for i in range(4):
            loc_layers.append(nn.Conv2d(64, 4, kernel_size=1))
        self.loc_layers = loc_layers

    @torch.jit.script_method
    def forward(self, x):
        return x

class MultiBox2(ScriptModule):
    def __init__(self):
        super().__init__()
        loc_layers = nn.ModuleList()
        for i in range(4):
            loc_layers.append(nn.Conv2d(64, 4, kernel_size=1))
        self.loc_layers = loc_layers

    @torch.jit.script_method
    def forward(self, x):
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = MultiBox()  # workaround approach
        self.model2 = MultiBox2()  # same as model1, demonstrating encapsulation

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return torch.tensor(torch.allclose(out1, out2), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 64, 32, 32, dtype=torch.float32)

