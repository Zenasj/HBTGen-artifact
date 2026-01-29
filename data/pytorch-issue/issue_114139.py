# torch.randint(10, 30, (B,), dtype=torch.int64)  # B is batch size
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        batch_size = a.numel()
        max_len = a.max()
        range_tensor = torch.arange(0, max_len, device=a.device).type_as(a)
        expanded = range_tensor.repeat(batch_size, 1)
        comparison = expanded.lt(a.unsqueeze(1))
        return ~comparison

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randint(10, 30, (10,), dtype=torch.int64)
    a[0] = 29  # Fix first element to 29 as in original test
    return a

