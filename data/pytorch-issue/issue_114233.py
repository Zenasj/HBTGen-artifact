# torch.rand(3, 1, dtype=torch.float64, requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, input):
        dim = 0
        values, indices = torch.max(input, dim)
        return values

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduce the specific input values from the issue comment
    return torch.tensor([[1.3336e241], [1.3823e-310], [2.1646e-304]], 
                       dtype=torch.float64, requires_grad=True)

