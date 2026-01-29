# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        # The SVD operation may trigger autograd fallback leading to CppFunctionTensorPreHook error
        U, S, Vh = torch.linalg.svd(x)
        return U

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    return torch.rand(B, 10, dtype=torch.float32)

