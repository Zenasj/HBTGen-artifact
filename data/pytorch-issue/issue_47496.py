import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32)
        ])

    def forward(self, x):
        i = 1
        x = self.layers[i](x)
        return x

torch.jit.script(Model())

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32)
        ])

    def forward(self, x, n_layers: int):
        for i in range(n_layers):
            layer: torch.nn.Linear = self.layers[i]
            x = layer(x)
        return x

torch.jit.script(Model())

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32)
        ])

    def forward(self, x):
        i = 1
        x = self.layers[i](x)   # failed
        return x

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10)] * 3)
        self.z = tuple([0, 1, 2])

    def forward(self, x):
        x = self.layers[0](x)
        for a in self.z[1:]:
            x = self.layers[a](x)
        return x

from torch.nn import ModuleList
import torch 

@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass

class Mine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)])

    def forward(self, x: torch.Tensor, c: int):
        for i in range(c):
            submodule: ModuleInterface = self.layers[i]
            result = submodule.forward(x)
            return result 

torch.jit.script(Mine())