# torch.rand(B, 2, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Sequential):
    def __add__(self, other):
        if not isinstance(other, MyModel):
            raise TypeError(f"unsupported operand type(s) for +: {type(self)} and {type(other)}")
        # Combine modules from both Sequentials
        combined_modules = list(self.children()) + list(other.children())
        return MyModel(*combined_modules)

def my_model_function():
    # Create two Sequential parts and combine them using __add__
    part1 = MyModel(nn.Linear(2, 3), nn.ReLU())
    part2 = MyModel(nn.Linear(3, 1))
    return part1 + part2  # Returns a new MyModel instance

def GetInput():
    # Input tensor matching the first layer's input size (2 features)
    return torch.rand(5, 2)  # Batch size 5, 2 input features

