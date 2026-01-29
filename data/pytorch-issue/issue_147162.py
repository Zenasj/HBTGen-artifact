import inspect
import torch
from torch import nn

def greet(greeting, name, punctuation='!'):
    """Simple function to greet a person."""
    print(f"{greeting}, {name}{punctuation}")

# torch.rand(3)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = inspect.signature(greet)
    
    def forward(self, x):
        self.sig.bind("Hello", "Alice")  # Dynamo graph break occurs here
        return torch.sin(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)

