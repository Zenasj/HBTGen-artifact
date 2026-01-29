# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class ScaleFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input, scale)
        return input * scale

    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors  # Use ctx.saved_tensors for PyTorch 0.4+
        grad_input = grad_output * scale
        grad_scale = torch.mean(grad_output * input)
        return grad_input, grad_scale

class MyModel(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        # Use torch.full((), init_value) to create a scalar (0-dim) parameter
        self.scale = nn.Parameter(torch.full((), init_value))

    def forward(self, input):
        return ScaleFunc.apply(input, self.scale)

def my_model_function():
    return MyModel()

def GetInput():
    # Match the original example's input shape (3,7) extended to 4D (B,C,H,W)
    return torch.randn(3, 7, 1, 1)

