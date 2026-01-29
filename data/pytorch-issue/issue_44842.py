# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Example input shape for a typical model
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._prop = None  # Property storage
    
    @property
    def prop(self):
        return self._prop
    
    @prop.setter
    def prop(self, val):
        # Reproduces the problematic string formatting in setter
        some_long_continued_string = f"""\
        dedent here lol
        """
        self._prop = some_long_continued_string
    
    def forward(self, x):
        # Dummy forward pass to satisfy nn.Module requirements
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

