# torch.rand(1, dtype=torch.float32)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.softplus = nn.Softplus(beta=3, threshold=1)  # As specified in the original issue's code

    def forward(self, x):
        return self.softplus(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32, requires_grad=True)  # Matches input shape and requires_grad=True

