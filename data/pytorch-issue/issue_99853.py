# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        start, end, step, epsilon = x.unbind()
        end_add = end + epsilon
        end_sub = end - epsilon

        arr_add = torch.arange(start, end_add, step)
        arr_sub = torch.arange(start, end_sub, step)

        shape_diff = torch.tensor(arr_add.shape[0] != arr_sub.shape[0], dtype=torch.bool)
        element_diff = torch.any(arr_add != arr_sub)
        total_diff = shape_diff | element_diff
        return total_diff.unsqueeze(0)

def my_model_function():
    return MyModel()

def GetInput():
    start = torch.tensor(0.0)
    end = torch.rand(())
    step = torch.rand(()) * 0.1 + 0.01
    epsilon = torch.tensor(1e-8)
    return torch.stack([start, end, step, epsilon])

