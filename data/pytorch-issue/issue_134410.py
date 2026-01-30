import torch
import torch.nn as nn

class CubicRootSmoother(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input != 0.0,  (input / (1 + torch.abs(input)**3)**(1/3)), 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return  torch.where(input != 0.0,  grad_output * ((((((1 + ((torch.abs(input)**3)))**(1/3))) - ((input/3)* ((3*((input/torch.abs(input))))**(-1/3)))) / (( ((1 + ((torch.abs(input)**3)))**(1/3))**2)))), 0.0)

class CubicRootSmootherModule(nn.Module):
    def forward(self, input):
        return CubicRootSmoother.apply(input)