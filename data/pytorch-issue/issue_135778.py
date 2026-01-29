# Input shape: three tensors of (8, 64, 1024, 1024), dtype=torch.float16
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        q, k, v = inputs
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def my_model_function():
    return MyModel()

def GetInput():
    shape = (8, 64, 1024, 1024)
    q = torch.randn(shape, dtype=torch.float16, device="cuda")
    k = torch.randn(shape, dtype=torch.float16, device="cuda")
    v = torch.randn(shape, dtype=torch.float16, device="cuda")
    return (q, k, v)

