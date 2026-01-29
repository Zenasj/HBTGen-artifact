# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, A):
        return torch.linalg.norm(A, ord=1.0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

