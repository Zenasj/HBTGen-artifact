# torch.rand(B, 3, dtype=torch.float32) for each input tensor in the tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        a, b = x  # x is a tuple of two tensors
        return a.shape[0] * a * b

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randn(4, 3)
    b = torch.randn(4, 3)
    torch._dynamo.mark_dynamic(a, 0)
    torch._dynamo.mark_dynamic(b, 0)
    return (a, b)

