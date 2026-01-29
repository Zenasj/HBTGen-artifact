import torch
import torch.nn as nn
import torch.nn.functional as F

# Input is a tuple of three tensors each with shape (3, 592, 4, 49, 32), dtype=torch.float32
class MyModel(nn.Module):
    def forward(self, inputs):
        q, k, v = inputs
        return F.scaled_dot_product_attention(q, k, v)

def my_model_function():
    return MyModel()

def GetInput():
    q = torch.rand(3, 592, 4, 49, 32)
    k = torch.rand(3, 592, 4, 49, 32)
    v = torch.rand(3, 592, 4, 49, 32)
    return (q, k, v)

