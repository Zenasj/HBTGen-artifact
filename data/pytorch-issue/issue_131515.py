# torch.rand(B, 4, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = x.clone()
        dims = [0, 1]
        shape = [x.shape[d] for d in dims]
        shape.append(1)
        return torch.ops.prims.broadcast_in_dim.default(x, shape, dims).clone()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4)

