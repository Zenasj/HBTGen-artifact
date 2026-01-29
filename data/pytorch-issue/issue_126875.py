# torch.rand(B, 10)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cst = nn.Parameter(torch.zeros(()))
        self.linear = nn.Linear(10, 10)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.__dict__["_modules"]["linear"], name)
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, input):
        # Working path using direct access to linear's attributes
        return F.linear(input, self.linear.weight, self.linear.bias) + self.cst

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10)  # Matches the input shape expected by MyModel's forward

