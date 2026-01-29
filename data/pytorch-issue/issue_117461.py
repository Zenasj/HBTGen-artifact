import torch
import torch.nn as nn

# torch.rand((), dtype=torch.int64)  # Input is a 0D integer tensor
class MyModel(nn.Module):
    def forward(self, x):
        y = x.item()
        return torch.ones(y).sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random integer tensor (0D) between 1 and 10
    return torch.randint(1, 10, (), dtype=torch.int64)

