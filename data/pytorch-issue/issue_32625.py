# torch.rand(100, dtype=torch.float32)  # Input shape inferred from example
import torch
from torch import nn, autograd

class Foo(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Clone to avoid view-based storage sharing (fixes the error)
        s = x.clone()[:10]
        ctx.save_for_backward(x, s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        # Dummy gradient for demonstration (replace with actual logic)
        x, s = ctx.saved_tensors
        return grad_output.clone()

class MyModel(nn.Module):
    def forward(self, x):
        return Foo.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, requires_grad=True)

