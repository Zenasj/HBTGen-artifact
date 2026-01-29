# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn
import math

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class MyModel(nn.Module):
    def __init__(self, out_channels):
        super(MyModel, self).__init__()
        self.weights = nn.ParameterList()
        for _ in range(3):
            weight = nn.Parameter(torch.Tensor(out_channels, out_channels))
            uniform(out_channels, weight)
            self.weights.append(weight)

    def forward(self, x):
        for i in range(3):
            x = torch.matmul(self.weights[i], torch.sigmoid(x))
        return x

def my_model_function():
    return MyModel(2)

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32)

