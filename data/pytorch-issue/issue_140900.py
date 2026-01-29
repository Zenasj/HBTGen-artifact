# torch.rand(B)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net2 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.net2(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Matches 1D input used in original test case

