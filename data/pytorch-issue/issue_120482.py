# torch.rand(2), torch.rand(1), torch.empty(2, 1)  # Example input tensors for shape (2, 1)
import torch
import random
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, tensors):
        input_t, vec_t, out_t = tensors
        return torch.outer(input_t, vec_t, out=out_t)

def my_model_function():
    return MyModel()

def GetInput():
    vec1 = random.randint(1, 10)  # Random size for first dimension
    vec2 = 1  # vec2 must be 1 to trigger the issue
    input_t = torch.randn(vec1)
    vec_t = torch.randn(vec2)
    out_t = torch.empty(vec1, vec2)
    return (input_t, vec_t, out_t)

