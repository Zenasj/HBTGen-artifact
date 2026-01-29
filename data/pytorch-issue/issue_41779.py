# torch.rand(3, 3, dtype=torch.float)

import torch
import torch.nn as nn

class CorrectedInfNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        norm_val = input.abs().max()
        ctx.save_for_backward(input)
        return norm_val

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        max_abs = input.abs().max()
        mask = (input.abs() == max_abs).float()
        num_max = mask.sum()
        grad_input = grad_output * input.sign() * mask / num_max
        return grad_input

class MyModel(nn.Module):
    def forward(self, x):
        current_out = torch.norm(x, p=float('inf'))
        corrected_out = CorrectedInfNorm.apply(x)
        return current_out, corrected_out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([
        [9., 2., 9.],
        [-2., -3., -4.],
        [7., 8., -9.],
    ], requires_grad=True)

