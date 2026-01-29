# torch.rand(3, 3, dtype=torch.float32), torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        A, B = inputs
        correct = A @ B
        buggy = A @ B.t()  # Simulates MPS bug (transposed B)
        return correct - buggy

def my_model_function():
    return MyModel()

def GetInput():
    N = 3
    A = torch.rand(N, N, dtype=torch.float32) * 2 - 1
    B = torch.rand(N, N, dtype=torch.float32) * 5 - 2.5
    return (A, B)

