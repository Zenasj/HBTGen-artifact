# torch.rand(1, 129, 1024, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1024, 2816, bias=False)  # Matches Linear(1024→2816, no bias)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    return model

def GetInput():
    # Returns a tensor with shape (1, 129, 1024) to replicate x2 in the original issue
    return torch.rand(1, 129, 1024, dtype=torch.float32)

