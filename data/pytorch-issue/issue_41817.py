# torch.rand(2, 2, dtype=torch.float32), torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, tensors):
        return torch.chain_matmul(*tensors)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2, 2, dtype=torch.float32), torch.rand(2, 2, dtype=torch.float32))

