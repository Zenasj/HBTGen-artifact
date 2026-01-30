import torch

class UseNeedsInputGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            return grad_output
        return None

def linear(x):
    return UseNeedsInputGradFunction.apply(x)

linear = torch.compile(linear)
x = torch.randn(4, 128, requires_grad=True)
y = linear(x)