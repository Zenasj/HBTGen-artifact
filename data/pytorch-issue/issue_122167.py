import torch
from torch import nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, x):
        return MySin.apply(x)

class MySin(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x.sin()

    @staticmethod
    def setup_context(*args, **kwargs):
        pass  # No context needed for this example

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.stride(0) > 1:
            return grad_output.sin()
        else:
            return grad_output.cos()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

