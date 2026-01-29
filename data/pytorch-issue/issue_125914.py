# torch.rand(2, dtype=torch.int64)  # Input shape is (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        u0, u1 = x.unbind()
        sum_val = u0 + u1
        max_val = torch.maximum(torch.tensor(1, dtype=torch.int64), sum_val)
        return (max_val == 20).to(torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([10, 10], dtype=torch.int64)

