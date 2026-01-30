import torch

class CustomAdd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        return x + 0

    @staticmethod
    def symbolic(g, x, y):
        return g.op('domain::CustomAdd', x, y)