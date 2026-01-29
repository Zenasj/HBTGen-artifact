# Input: tuple of 3 tensors each of shape (1, 5), dtype=torch.float32

import torch
from torch import nn

class TestFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensors):
        return tensors

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Increment gradients by 1 as per original example
        return (tuple(g + 1 for g in grad_outputs),)

class MyModel(nn.Module):
    def forward(self, tensors):
        return TestFunc.apply(tensors)

def my_model_function():
    return MyModel()

def GetInput():
    # Create 3 tensors of shape (1,5) with requires_grad enabled
    return (
        torch.rand(1, 5, requires_grad=True),
        torch.rand(1, 5, requires_grad=True),
        torch.rand(1, 5, requires_grad=True),
    )

