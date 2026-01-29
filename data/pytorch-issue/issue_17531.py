# torch.randint(10, (), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, num_iter: int):
        x = torch.ones([2, 2], dtype=torch.float32)
        v = torch.ones(2, 1, dtype=torch.float32)
        for i in range(num_iter):
            v = x * v
        return x, v

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(10, (), dtype=torch.int64)

