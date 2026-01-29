# (torch.rand(2, 2, dtype=torch.float32), torch.rand(2, 2, dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    def hook(t):
        t.grad.mul_(5)  # Original hook function from the issue
    x = torch.rand(2, 2, requires_grad=True)
    y = torch.rand(2, 2, requires_grad=True)
    x.register_post_accumulate_grad_hook(hook)
    return (x, y)

