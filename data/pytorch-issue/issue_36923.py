# torch.rand((), dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = 0  # Fixed t as per the example's condition

    def forward(self, x):
        y = self.t * (x / self.t)  # Creates NaNs in y when t is 0
        z = torch.where(x >= self.t, x, y)
        return z

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float32)

