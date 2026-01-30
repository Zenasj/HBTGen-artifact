import torch

class Alias(Function):
    @staticmethod
    def forward(ctx, x):
        return x[:]

    @staticmethod
    def backward(ctx, gx):
        return gx

inp = torch.rand(2, requires_grad=True)

with torch.no_grad():
    # Used to error out
    output = Alias.apply(inp)