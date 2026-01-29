# torch.rand(16, 4096, 40, dtype=torch.float), torch.rand(16, 4096, 40, dtype=torch.float), torch.rand(16, 4096, 40, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        q, k, v = inputs
        attn = torch.einsum('b i d, b j d -> b i j', q, k)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    q = torch.rand(16, 4096, 40, dtype=torch.float)
    k = torch.rand(16, 4096, 40, dtype=torch.float)
    v = torch.rand(16, 4096, 40, dtype=torch.float)
    return (q, k, v)

