# torch.rand(1, dtype=torch.float32)  # Dummy input of shape (1,)
import torch
from torch import nn
from typing import List

class AttributeAccumulation_Broken(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.array = torch.jit.Attribute([], List[int])

    def forward(self):
        self.array.append(0)
        print(self.array)
        return self.array

class AttributeAccumulation_Fixed(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.array = torch.jit.Attribute([], List[int])

    @torch.jit.script_method
    def forward(self):
        self.array.append(0)
        print(self.array)
        return self.array

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.broken = AttributeAccumulation_Broken()
        self.fixed = AttributeAccumulation_Fixed()

    def forward(self, x):
        broken_out = self.broken()
        fixed_out = self.fixed()
        # Compare outputs (broken returns [], fixed returns [0])
        return torch.tensor(broken_out == fixed_out, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input as required

