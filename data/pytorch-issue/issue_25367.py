# torch.rand(B, 3, dtype=torch.float)
import torch
from torch import nn

class ParamModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(size))

class TestModA(nn.Module):  # Original ParameterList-based model (buggy with JIT)
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterList([
            nn.Parameter(torch.zeros(3)),
            nn.Parameter(torch.zeros(3))
        ])
    
    def forward(self, x):
        return x + self.params[0] + self.params[1]

class TestModB(nn.Module):  # Workaround using ModuleList of Parameter-containing modules
    def __init__(self):
        super().__init__()
        self.params = nn.ModuleList([
            ParamModule(3),
            ParamModule(3)
        ])
    
    def forward(self, x):
        return x + self.params[0].param + self.params[1].param

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = TestModA()  # Problematic ParameterList variant
        self.model_b = TestModB()  # Valid workaround
        
    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        # Return comparison result as tensor (1.0 if outputs match, 0.0 otherwise)
        return torch.tensor(torch.allclose(out_a, out_b), dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float)  # Batch size 2, 3 features

