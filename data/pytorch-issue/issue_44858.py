# (torch.randint(-2, 3, (2,), dtype=torch.int32), torch.randint(1, 3, (2,), dtype=torch.int32))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        pt_result = torch.floor_divide(a, b)
        np_like = (a.float() / b.float()).floor().to(a.dtype)
        return torch.ne(pt_result, np_like)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randint(-2, 3, (2,), dtype=torch.int32)
    b = torch.randint(1, 3, (2,), dtype=torch.int32)
    return (a, b)

