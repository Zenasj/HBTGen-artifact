# (torch.rand(5,5), torch.rand(5,5))  # Input tuple for MyModel
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        pred_source, a_source = inputs
        pred = pred_source > 0
        a = a_source.t()
        out = torch.where(pred, a, 0)
        expected_stride = torch.tensor([5, 1])
        actual_stride = torch.tensor(out.stride())
        return torch.all(expected_stride == actual_stride)

def my_model_function():
    return MyModel()

def GetInput():
    pred_source = torch.rand(5, 5)
    a_source = torch.rand(5, 5)
    return (pred_source, a_source)

