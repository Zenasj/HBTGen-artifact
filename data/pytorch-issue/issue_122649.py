# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class X:
    _OUTPUTS = {"a": (torch.tensor(1),), "b": (torch.tensor(2),)}
    @property
    def outputs(self):
        return self._OUTPUTS["a"]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = X()  # Custom class with problematic __getitem__ usage
    
    def forward(self, input):
        # Reproduces the Dynamo error scenario
        val = self.x.outputs[0]  # getattr + getitem on custom class
        return val + input  # Example operation to keep forward valid

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

