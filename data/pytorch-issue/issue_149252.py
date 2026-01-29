# torch.rand(3, dtype=torch.float)  # Input shape is a 1D tensor of size 3
import torch
from torch import nn

class CustomRepeatInterleave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, repeats):
        ctx.repeats = repeats
        output = input.repeat_interleave(repeats)
        ctx.mark_dirty(output)  # Issue's problematic line
        return output

    @staticmethod
    def backward(ctx, grad_output):
        repeats = ctx.repeats
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        for i in range(repeats):
            grad_input += grad_output[i]  # As per user's fixed code
        return grad_input, None

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        repeats = 2
        return CustomRepeatInterleave.apply(x, repeats)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float)  # Matches the example's input shape

