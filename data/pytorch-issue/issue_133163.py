# torch.rand(1, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bytes_data = b'hello'  # Problematic bytes attribute

    def __getstate__(self):
        # Include non-tensor attributes in the saved state
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def forward(self, x):
        return x  # Dummy forward pass

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10)  # Input matching forward() expectations

