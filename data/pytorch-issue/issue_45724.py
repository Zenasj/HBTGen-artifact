# torch.rand(13269, 8, 22, 64, dtype=torch.float16) for query and key layers
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        query, key = inputs
        key_transposed = key.transpose(-1, -2)
        return torch.matmul(query, key_transposed)

def my_model_function():
    return MyModel()

def GetInput():
    query = torch.rand(13269, 8, 22, 64, dtype=torch.float16, device='cuda')
    key = torch.rand(13269, 8, 22, 64, dtype=torch.float16, device='cuda')
    return (query, key)

