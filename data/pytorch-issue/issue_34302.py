# torch.rand(1, dtype=torch.float32)  # Assuming a simple scalar input for demonstration

import torch
import torch.nn as nn

class Sub(nn.Module):
    def __init__(self, i):
        super(Sub, self).__init__()
        self.i = i

    def forward(self, thing):
        return thing - self.i

class MyModel(nn.Module):
    __constants__ = ['mods']

    def __init__(self):
        super(MyModel, self).__init__()
        self.mods = nn.ModuleList([Sub(i) for i in range(10)])

    def forward(self, v):
        v = self.mods[4].forward(v)
        v = self.mods[-1].forward(v)
        v = self.mods[-9].forward(v)
        return v

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, dtype=torch.float32)

