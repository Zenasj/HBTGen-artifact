# torch.rand((), dtype=torch.float32)
from dataclasses import GenericAlias, _FIELDS
import torch
from torch import nn

def is_dataclass_recode(obj):
    cls = obj if isinstance(obj, type) and not isinstance(obj, GenericAlias) else type(obj)
    return hasattr(cls, _FIELDS)

class MyModel(nn.Module):
    def forward(self, x):
        if not is_dataclass_recode(x):
            return x + 1
        else:
            return x  # Fallback to return input if dataclass is detected

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float32)

