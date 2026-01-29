# Input is a tuple: (torch.rand(6, dtype=torch.float32), torch.randint(0, 3, (6,), dtype=torch.long), torch.rand(4, dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, args):
        x, index, input = args
        y_max = input.scatter_reduce(0, index, x, reduce="amax")
        y_sum = input.scatter_reduce(0, index, x, reduce="sum")
        y_min = input.scatter_reduce(0, index, x, reduce="amin")
        y_mul = input.scatter_reduce(0, index, x, reduce="prod")
        return y_max, y_sum, y_min, y_mul

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(6, dtype=torch.float32)
    index = torch.randint(0, 3, (6,), dtype=torch.long)  # Matches original index range (0-2)
    input_tensor = torch.rand(4, dtype=torch.float32)
    return (x, index, input_tensor)

