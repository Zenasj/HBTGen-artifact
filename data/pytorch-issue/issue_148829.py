import torch.nn as nn

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class Linear(nn.Linear):
    pass

    # def forward(self, x):
    #     y = super().forward(x)
    #     return x + y
    
class Test(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Linear(10, 10, device="cuda") for i in range(10)])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = checkpoint(self.layers[i], x)
        return x
    
model = Test()
def compile(model):
    return torch.compile(model, mode="max-autotune")
for i in range(len(model.layers)):
    model.layers[i] = compile(model.layers[i])
x = torch.randn((10, 10), device="cuda", requires_grad=True)

y = model(x)

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False

class Test(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10, device="cuda") for i in range(10)])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = checkpoint(self.layers[i], x) + x
        return x
    
model = Test()
def compile(model):
    return torch.compile(model, mode="max-autotune")
for i in range(len(model.layers)):
    model.layers[i] = compile(model.layers[i])
x = torch.randn((10, 10), device="cuda", requires_grad=True)

y = model(x)
y.sum().backward()