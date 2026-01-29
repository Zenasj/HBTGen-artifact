# torch.rand(B, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Example 1: view and in-place add
        tmp1 = torch.ones_like(x)
        y = x.view_as(tmp1)
        y.add_(tmp1)
        
        # Example 2: in-place add on temporary
        tmp2 = torch.ones_like(x)
        tmp2.add_(x)
        
        return y + tmp2

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (B=2, 4 elements)
    return torch.rand(2, 4)  # Batch size 2 as in test examples

