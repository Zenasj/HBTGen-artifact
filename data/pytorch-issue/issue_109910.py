import torch
import torch.nn as nn

class dtype_test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dtype=torch.float32):
        orig_precision = tensor.dtype
        ctx.save_for_backward(tensor)
        ctx.orig_precision = orig_precision
        return tensor.to(dtype), dtype  # Returns tensor and dtype

    @staticmethod
    def backward(ctx, grad_tensor, grad_dtype):
        # Handle gradients: only the tensor output contributes to gradient
        return grad_tensor.to(ctx.orig_precision), None

class MyModel(nn.Module):
    def __init__(self, target_dtype=torch.bfloat16):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x):
        # Use only the tensor output from autograd function
        converted, _ = dtype_test.apply(x, self.target_dtype)
        return converted

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape matching the repro example
    return torch.randn(16, 16, dtype=torch.float32)

