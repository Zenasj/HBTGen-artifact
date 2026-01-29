# torch.rand(1, 1, 2, 2, dtype=torch.float32)  # Input shape (B=1, C=1, H=2, W=2) inferred from example's 2x2 tensor usage

import torch
from torch import nn

class Submodule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x + 2

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule1 = Submodule()
        self.submodule2 = torch.jit.script(Submodule())

    @torch.jit.ignore
    def some_debugging_function(self, x):
        # Placeholder for non-scriptable debugging code (e.g., pdb)
        pass

    @torch.jit.ignore(drop_on_export=True)
    def training_only_code(self, x):
        # Simulate non-scriptable training code (returns zeros to avoid runtime errors)
        return torch.zeros_like(x)

    @torch.jit.export
    def an_explicit_entry_point(self, x):
        return self.forward(x + 20)

    @torch.jit.export
    def a_called_model_entry_point(self, x):
        return self.forward(x + 20)

    def some_function(self, y):
        return y + 25

    def forward(self, x):
        x += self.submodule1(x)
        x += self.submodule2(x)
        x += self.some_function(x)
        if self.training:
            x += self.training_only_code(x)
        self.some_debugging_function(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 2, 2, dtype=torch.float32)

