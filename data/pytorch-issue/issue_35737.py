# torch.rand(2, 3, 4), torch.rand(2, 6, 4) ← Input tensors B and A
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        B, A = inputs  # B shape: (2,3,4), A shape: (2,6,4)
        return torch.einsum("noq,npq->nop", B, A)

def my_model_function():
    return MyModel()

def GetInput():
    B = torch.rand(2, 3, 4)
    A = torch.rand(2, 6, 4)
    return (B, A)  # Returns tuple (B, A) matching the einsum equation's input requirements

