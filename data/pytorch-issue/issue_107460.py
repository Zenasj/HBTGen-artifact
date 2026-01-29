# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class BinaryStatScores(nn.Module):
    def __init__(self):
        super().__init__()

class Accuracy(nn.Module):
    def __new__(cls):
        return BinaryStatScores()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        y = Accuracy()
        return self.layer(x)

def overwrite_torch_functions():
    module_set_attr_orig = torch.nn.Module.__setattr__

    def wrap_set_attr(self, name, value):
        if isinstance(value, torch.nn.Module):
            print(value)  # <-- calls __repr__ on the module
        module_set_attr_orig(self, name, value)

    torch.nn.Module.__setattr__ = wrap_set_attr

def my_model_function():
    overwrite_torch_functions()
    return MyModel()

def GetInput():
    return torch.rand(2, 2)

