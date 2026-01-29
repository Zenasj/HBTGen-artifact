# torch.randint(0, 1e9, (), dtype=torch.int64)
import torch
from torch import nn

class CPUModule(nn.Module):
    def forward(self, x):
        return x % 2

class GPUModule(nn.Module):
    def forward(self, x):
        return torch.fmod(x.float(), 2).int()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpu_mod = CPUModule()
        self.gpu_mod = GPUModule()
    
    def forward(self, x):
        cpu_out = self.cpu_mod(x)
        gpu_out = self.gpu_mod(x)
        return torch.all(torch.eq(cpu_out, gpu_out))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(421123111, dtype=torch.int64)

