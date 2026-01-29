# Input is a tuple of four tensors (q, k, v, bias) each of shape (1, 1, 16, 16) with dtype=torch.float16
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        q, k, v, bias = inputs
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, bias)

def my_model_function():
    return MyModel()

def GetInput():
    q = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")
    k = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")
    v = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")
    bias = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")
    return (q, k, v, bias)

