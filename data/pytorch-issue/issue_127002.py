# torch.rand(1, 1, 2, 2, dtype=torch.float16), torch.rand(1, 1, 1, 2, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        input, grid = x
        return torch.nn.functional.grid_sample(input, grid, align_corners=False)

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.rand(1, 1, 2, 2, dtype=torch.float16)
    grid = torch.rand(1, 1, 1, 2, dtype=torch.float16)
    return (input, grid)

