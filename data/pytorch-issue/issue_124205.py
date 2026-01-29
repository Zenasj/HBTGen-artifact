# Input is a NestedTensor with values of shape (N,) and offsets of shape (B+1,). Example: values=torch.randn(20), offsets=torch.tensor([0, 2, 4, ..., 20])
from torch.nested._internal.nested_tensor import NestedTensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x: NestedTensor) -> NestedTensor:
        x = x.clamp(0.1, 0.5)
        x *= x._max_seqlen  # Problematic line with static specialization
        return x

def my_model_function():
    return MyModel()

def GetInput():
    el_per_row = 2  # Example parameter for input generation
    torch.manual_seed(0)
    values = torch.randn(10 * el_per_row, device="cuda")
    offsets = torch.arange(11, device="cuda") * el_per_row
    return NestedTensor(values, offsets)

