# (torch.rand(2, dtype=torch.bool), torch.rand(2, dtype=torch.bool), torch.rand(2, dtype=torch.bool))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        condition, x, y = inputs
        # Convert BoolTensors to int to bypass missing where implementation for bool
        x_int = x.to(torch.int)
        y_int = y.to(torch.int)
        result_int = torch.where(condition, x_int, y_int)
        return result_int.to(torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random BoolTensors of shape (2,)
    condition = torch.rand(2) > 0.5
    x = torch.rand(2) > 0.5
    y = torch.rand(2) > 0.5
    return (condition, x, y)

